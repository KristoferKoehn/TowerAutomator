import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import get_close_matches
from enum import Enum

import cv2
import numpy as np

import MouseController
from MouseController import get_mouse_controller
from TemporalValueNormalizer import TemporalValueNormalizer
from debug_draw_manager import debug_draw_manager
from debug_logging_manager import get_debug_logger
from kutils import load_templates, find_button_center, window, Menu, load_image_from_path, clean_ocr_string
from ocr_worker import ocr_pool
from tracker.strategy_base import SubStrategy
from PIL import Image


class UpgradeSearcherStrategy(SubStrategy):
    """
    Scans the bottom bar for visible upgrades and clicks them.
    """
    upgrade_templates = load_templates("upgrade_template")
    defense_button_template = load_image_from_path("ui_templates/defense_menu_button.png")
    attack_button_template = load_image_from_path("ui_templates/attack_menu_button.png")
    utility_button_template = load_image_from_path("ui_templates/utility_menu_button.png")

    attack_upgrade_template = load_image_from_path("ui_templates/attack_upgrades.png")
    defense_upgrade_template = load_image_from_path("ui_templates/defense_upgrades.png")
    utility_upgrade_template = load_image_from_path("ui_templates/utility_upgrades.png")

    result_lock = threading.Lock()
    resource_lock = threading.Lock()

    latest_upgrades_detected = []
    tracked_upgrades = {}

    lowest_known_cost_item = ""

    current_menu = Menu.ATTACK

    rejected_names = [
        "RAPIDCHANCEFIRE", "DAMAGETHORN", "LANDDAMAGEMINE", "LANDCHANCEMINE",
        "LANDNINECHANCE", "RAPIDDURATIONFIRE", "LANDRADIUSMINE",  "COINSJAVE",
        "LANDNINERADIUS", "LANDNINEDAMAGE", "NAXRECOVERY", "COINSKCILLBONUS",
        "INTERESTJAVE", "CASHBONUS", "CASHLAVE", "RECOVERYMAX", "DEFENSE00",
        "COINSBONUSKILL", "SHOCKIAVESIZE", "BOUNCESHOTCHANCE", "ATTACKUPGRADEFREE",
        "PACKAGECHANCE", "NULTISHOTCHANCE", "NULTISHOTTARGETS", "NHULTISHOTTARGETS",
        "BOUNCESHOTTARGETS","AOSOLUTE", "KEGEN", "DEFENSEOO", "RORCE", "FORCE", "None",
        "SHECKWZVESIZE", "SHECKWZVEFREQUENCY", "SHOCKWAVEFREQUETCY", "SHECKWAVESIZE", "INTERESTWEAVE",
        "FREEUTILITYUPGRZDE" "TCRITICALFACTOR", "IVAXRECOVERY", "HEAITH", "ABSOLUTE" "MMAXRECOVERY",
        "DEFENSEABSOIUTE", "ABSOLUTE", "SHOCKIVAVE"
    ]

    # Normalized target names for fuzzy matching
    normalized_upgrade_names = [
        "DAMAGE", "ATTACKSPEED", "CRITICALCHANCE", "CRITICALFACTOR", "RANGE", "DAMAGEPERMETER",
        "MULTISHOTCHANCE", "MULTISHOTTARGETS", "RAPIDFIRECHANCE", "RAPIDFIREDURATION", "BOUNCESHOTCHANCE",
        "BOUNCESHOTTARGETS", "BOUNCESHOTRANGE", "SUPERCRITCHANCE", "SUPERCRITMULT", "RENDARMORCHANCE",
        "RENDARMORMULT", "HEALTH", "HEALTHREGEN", "DEFENSE", "DEFENSEABSOLUTE", "THORNDAMAGE", "LIFESTEAL",
        "KNOCKBACKCHANCE", "KNOCKBACKFORCE", "ORBSPEED", "ORBS", "SHOCKWAVESIZE", "SHOCKWAVEFREQUENCY",
        "LANDMINECHANCE", "LANDMINEDAMAGE", "LANDMINERADIUS", "DEATHDEFY", "WALLHEALTH", "WALLREBUILD",
        "CASHBONUS", "CASHWAVE", "COINSKILLBONUS", "COINSWAVE", "FREEATTACKUPGRADE", "FREEDEFENSEUPGRADE",
        "FREEUTILITYUPGRADE", "INTERESTWAVE", "RECOVERYAMOUNT", "MAXRECOVERY", "PACKAGECHANCE",
        "ENEMYATTACKLEVELSKIP", "ENEMYHEALTHLEVELSKIP"
    ]

    current_resources_detected = ("", "", "")

    cash_value_normalizer = TemporalValueNormalizer(max_age=5.0)
    coin_value_normalizer = TemporalValueNormalizer(max_age=5.0)
    gem_value_normalizer = TemporalValueNormalizer(max_age=5.0)

    def matches(self, screenshot) -> bool:
        return False

    def __init__(self):
        self.screenshot = None
        self._stop_threads = threading.Event()
        self._stop_threads.set()
        self.upgrade_thread = threading.Thread(target=self.upgrade_detection_worker, daemon=True).start()
        self.resource_thread = threading.Thread(target=self.resource_detection_worker, daemon=True).start()
        self.menu_upgrade_lockout_timer = time.time() + 2.0

        self.scroll_timer = time.time()
        self.scroll_count = 0
        self.swap_menu_flag = False
        self.delay_timer_started_at = 0
        self.delayed_upgrade_check_pending = False
        self.wander_timer = time.time()


    def on_enter(self):
        get_debug_logger().log_with_spinner_until("[🛠️ UpgradeSearcher] Ready to search.")
        get_debug_logger().log_with_spinner_until("upgrade data thread started")
        self.menu_upgrade_lockout_timer = time.time() + 1.5
        self._stop_threads.clear()
        self.scroll_count = 0
        self.tracked_upgrades = {}
        self.delay_timer_started_at = 0
        self.delayed_upgrade_check_pending = False
        self.wander_timer = time.time()


    def on_exit(self):
        get_debug_logger().log_with_spinner_until("[🛠️ UpgradeSearcher] Cleaning search state.")
        self._stop_threads.set()


    ### MAIN RUN FUNCTION!!! =================
    frame_count = 0

    def run(self, screenshot: Image.Image):
        dlm = get_debug_logger()

        self.screenshot = screenshot

        with self.resource_lock:
            self.cash_value_normalizer.add_money_value(self.current_resources_detected[0])
            self.coin_value_normalizer.add_money_value(self.current_resources_detected[1])
            self.gem_value_normalizer.add_value(self.current_resources_detected[2])

        dlm.resources = (self.cash_value_normalizer.get_mode_value(),
                         self.coin_value_normalizer.get_mode_value(),
                         self.gem_value_normalizer.get_smoothed_value())

        lowest_cost = float("inf")
        for k, v in self.tracked_upgrades.items():
            cost_val = v[1]
            if cost_val is None or cost_val == "MAX":
                continue
            try:
                cost_float = float(cost_val)

                if cost_float < lowest_cost:
                    lowest_cost = cost_float
                    self.lowest_known_cost_item = k

            except (ValueError, TypeError):
                get_debug_logger().log_with_spinner_until(f"Invalid cost value for {k}: {cost_val}")

        dlm.lowest_cost_upgrade = self.lowest_known_cost_item

        prev_menu = self.current_menu
        # detect menu state
        attack_menu = find_button_center(self.screenshot, self.attack_upgrade_template, "attack menu",
                                         threshold=0.85) if self.frame_count % 10 == 0 else None
        if attack_menu:
            self.current_menu = Menu.ATTACK
        defense_menu = find_button_center(self.screenshot, self.defense_upgrade_template, "defense menu",
                                          threshold=0.85) if self.frame_count % 10 == 4 else None
        if defense_menu:
            self.current_menu = Menu.DEFENSE
        utility_menu = find_button_center(self.screenshot, self.utility_upgrade_template, "utility menu",
                                          threshold=0.85) if self.frame_count % 10 == 8 else None
        if utility_menu:
            self.current_menu = Menu.UTILITY

        dlm.current_upgrade_menu = self.current_menu
        dlm.current_upgrades_detected = self.latest_upgrades_detected
        dlm.tracked_upgrades = self.tracked_upgrades

        if prev_menu != self.current_menu:
            self.menu_upgrade_lockout_timer = time.time()

        if time.time() - self.menu_upgrade_lockout_timer > 0.7:
            self.update_visible_upgrades()
        else:
            self.clear_visible_upgrades()

        with self.result_lock:
            if len(self.latest_upgrades_detected) == 0:
                self.scroll_timer = time.time()

        self.frame_count += 1
        if self.frame_count > 100 == 0:
            self.frame_count = 0

        with self.result_lock:
            if time.time() - self.wander_timer > 85:
                matched_upgrade = None
                for upgrade in self.latest_upgrades_detected:
                    try:
                        raw_name = upgrade[0]
                        if raw_name:
                            cleaned_name = self.match_upgrade_name(raw_name)
                            if cleaned_name == self.lowest_known_cost_item:
                                matched_upgrade = upgrade
                                break
                    except (IndexError, TypeError):
                        continue  # Skip malformed entries

                # Check if lowest upgrade is still on screen
                if matched_upgrade:
                    if not self.delayed_upgrade_check_pending:
                        self.delay_timer_started_at = time.time()
                        self.delayed_upgrade_check_pending = True
                    elif time.time() - self.delay_timer_started_at >= 0.25:
                        # After half a second, re-check its presence
                        if matched_upgrade:
                            self.delay_timer_started_at = time.time() - 0.3 #hopefully stop the spam? or maybe this is fine..
                            MouseController.touch_position(matched_upgrade[3][0],matched_upgrade[3][1])
                            return ## exit to prevent scrolling or something
                        else:
                            self.delayed_upgrade_check_pending = False
                else:
                    self.delayed_upgrade_check_pending = False

        # Movement logic proceeds as normal after delay
        if time.time() - self.scroll_timer > 2.5 and self.scroll_count < 6:
            self.scroll_down()
            self.scroll_timer = time.time()
            self.scroll_count += 1
        elif time.time() - self.scroll_timer > 2.5 and self.scroll_count >= 6:
            self.scroll_up()
            self.scroll_count = 0
            self.scroll_timer = time.time()
            self.swap_menu_flag = True

        if time.time() - self.wander_timer < 100:
            if time.time() - self.scroll_timer > 2 and self.swap_menu_flag:
                self.swap_menu_flag = False
                self.switch_menus()
        else:
            if time.time() - self.scroll_timer > 2 and self.swap_menu_flag:
                self.swap_menu_flag = False
                self.switch_menu_to(self.tracked_upgrades[self.lowest_known_cost_item][2])
                self.scroll_timer = time.time() # give it a little pause so it can see the stuff at the top of the list


    def upgrade_detection_worker(self):
        while True:
            if not self._stop_threads.is_set():
                try:
                    if self.screenshot is not None:
                        result = detect_upgrade_templates(self.screenshot, self.upgrade_templates)
                        # Safely update global variable
                        with self.result_lock:
                            self.latest_upgrades_detected = result
                except Exception as e:
                    with open("thread_errors.log", "a") as f:
                        f.write("Exception in upgrade_detection_worker:\n")
                        traceback.print_exc(file=f)
                    raise  # Re-raise to allow the thread to crash loudly if desired
            else:
                time.sleep(1)

    def resource_detection_worker(self):
        while True:
            if not self._stop_threads.is_set():
                try:
                    if self.screenshot is not None:
                        result = resource_info(self.screenshot)

                        # Safely update global variable
                        with self.resource_lock:
                            self.current_resources_detected = result
                except Exception as e:
                    with open("thread_errors.log", "a") as f:
                        f.write("Exception in resource_detection_worker:\n")
                        traceback.print_exc(file=f)
                    raise
            else:
                time.sleep(1)

    def clear_visible_upgrades(self):
        with self.resource_lock:
            self.latest_upgrades_detected.clear()

    def update_visible_upgrades(self):
        with self.resource_lock:
            for name, value, cost, position in self.latest_upgrades_detected:
                if len(name) > 0 and name is not None:
                    name = self.match_upgrade_name(name)
                    if name is not None:
                        self.tracked_upgrades[name] = (value, normalize_cost_value(cost), self.current_menu)

    def switch_menus(self):
        match self.current_menu:
            case Menu.ATTACK:
                menu = find_button_center(self.screenshot, self.defense_button_template,"defense_menu")
                if menu:
                    MouseController.touch_position(menu[0], menu[1])
            case Menu.DEFENSE:
                menu = find_button_center(self.screenshot, self.utility_button_template, "utility_menu")
                if menu:
                    MouseController.touch_position(menu[0], menu[1])
            case Menu.UTILITY:
                menu = find_button_center(self.screenshot, self.attack_button_template, "attack_menu")
                if menu:
                    MouseController.touch_position(menu[0], menu[1])

    def switch_menu_to(self, target_menu):
        if self.current_menu == target_menu:
            return
        match target_menu:
            case Menu.ATTACK:
                menu = find_button_center(self.screenshot, self.attack_button_template, "attack_menu")
                if menu:
                    MouseController.touch_position(menu[0], menu[1])
            case Menu.DEFENSE:
                menu = find_button_center(self.screenshot, self.defense_button_template, "defense_menu")
                if menu:
                    MouseController.touch_position(menu[0], menu[1])
            case Menu.UTILITY:
                menu = find_button_center(self.screenshot, self.utility_button_template, "utility_menu")
                if menu:
                    MouseController.touch_position(menu[0], menu[1])

    def scroll_down(self):
        #new resolution, each upgrade tile is 88pixels tall
        h, w = self.screenshot.shape[:2]
        #MouseController.swipe(w//2, int(h * .8), w//2, int(h * .8) - 88, 1100)
        MouseController.drag(w//2, int(h * .85), w//2, int(h * .85) - 93, 800)

    def scroll_up(self):
        # new resolution, each upgrade tile is 88pixels tall
        h, w = self.screenshot.shape[:2]
        MouseController.swipe(w // 2, int(h * .72), w // 2, int(h * .75) + 230, 300)


    def match_upgrade_name(self, ocr_string: str, cutoff=0.8) -> str | None:
        """
        Returns the best-matched upgrade name from normalized list using fast fuzzy matching.
        :param ocr_string: Raw string from OCR.
        :param cutoff: Similarity threshold (0.0–1.0), higher means stricter match.
        :return: Cleaned and matched upgrade name, or None if not found.
        """
        cleaned = clean_ocr_string(ocr_string, self.rejected_names)
        if cleaned is not None:
            matches = get_close_matches(cleaned, self.normalized_upgrade_names, n=1, cutoff=cutoff)
            return matches[0] if matches else None
        else:
            return None

def resource_info(image):
    crop_x, crop_y = 36, 52  # top-left corner of the crop
    crop_w, crop_h = 123, 139

    result = ("--", "--", "--")

    def draw(_image):
        cv2.rectangle(_image, (crop_x, crop_y), (crop_w, crop_h), (255, 0, 0), 2)

    debug_draw_manager.set_draw("money", draw)

    top_left_crop = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    ocr_result = ocr_pool.submit_and_wait(top_left_crop)

    # Sort OCR results by vertical position (top to bottom)
    ocr_result_sorted = sorted(ocr_result, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)

    if len(ocr_result_sorted) >= 3:
        label_tuple = tuple(text.strip() for (_, text, _) in ocr_result_sorted[:3])
        result = label_tuple

    return result

def detect_upgrade_templates(image, templates, threshold=0.65):
    if image is None:
        raise ValueError("Input image is None")

    results = []
    debug_keys_this_frame = []
    height, width = image.shape[:2]
    lower_third_start = int(height * 2 / 3)

    lower_third = image[lower_third_start:, :]

    search_sector = cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY)
    boxes = []
    confidences = []

    for template in templates:

        res = cv2.matchTemplate(search_sector, template[1], cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            full_y = pt[1] + lower_third_start
            box = [pt[0], full_y, 200, 80]  # x, y, w, h
            conf = float(res[pt[1], pt[0]])
            boxes.append(box)
            confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=0.6)
    if len(indices) > 0:

        ocr_results = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}

            for box_index, i in enumerate(indices.flatten()):
                x, y, w, h = boxes[i]
                roi = image[y:y + h, x:x + w]

                # Submit the OCR task, track by box_index
                future = executor.submit(ocr_pool.submit_and_wait, roi)
                futures[future] = box_index

            for future in as_completed(futures):
                box_index = futures[future]
                try:
                    ocr_results[box_index] = future.result()
                except Exception as e:
                    ocr_results[box_index] = f"Error: {e}"

        for box_index, i in enumerate(indices.flatten()):
            x, y, w, h = boxes[i]
            conf = confidences[i]

            color = (255, 255, 0)

            upgrade_name = ""
            value = ""
            cost = ""

            for text_index, (bbox, text, conf) in enumerate(ocr_results[box_index]):
                (tl, tr, br, bl) = bbox
                cx = int((tl[0] + br[0]) / 2)
                cy = int((tl[1] + br[1]) / 2)

                label = ""
                if cx < w / 2:
                    upgrade_name += text + " "
                    label = "name"
                else:
                    if cy < h / 2:
                        value = text
                        label = "value"
                    else:
                        cost = text
                        label = "cost"

                pt1 = (x + int(tl[0]), y + int(tl[1]))
                pt2 = (x + int(br[0]), y + int(br[1]))
                pt3 = (x + int(tl[0]), y + int(tl[1]) - 2)

                def draw_label(debug_image, _pt1=pt1, _pt2=pt2, _pt3=pt3, _conf=conf, _label=label):
                    cv2.rectangle(debug_image, _pt1, _pt2, (255, 0, 0), 1)
                    cv2.putText(debug_image, f"{_label}, {_conf:.2f}", _pt3, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                key_label = f"upgrade_label_{box_index}_{text_index}"
                debug_draw_manager.set_draw(key_label, draw_label)
                debug_keys_this_frame.append(key_label)

            results.append((upgrade_name.strip(), value.strip(), cost.strip(), (x + 160, y + 40)))
            def draw_box(debug_image, _x=x, _y=y, _w=w, _h=h, _color=color, _conf=conf, _upgrade_name=upgrade_name.strip()):
                cv2.rectangle(debug_image, (_x, _y), (_x + _w, _y + _h), _color, 2)
                cv2.putText(debug_image, f"{_upgrade_name}, {_conf:.2f}", (_x + 5, _y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.circle(debug_image, (_x + 160, _y + 40), 4, _color, 2)

            key_box = f"upgrade_box_{box_index}"
            debug_draw_manager.set_draw(key_box, draw_box)
            debug_keys_this_frame.append(key_box)

    debug_draw_manager.clear_unused_debug_draws(f"upgrade", debug_keys_this_frame)
    return results

def normalize_cost_value(s: str) -> str:
    if s.endswith('x'):
        return "MAX"
    else:
        parsed = parse_ocr_number(s[1:])
        return parsed

def parse_ocr_number(text: str) -> str | None:
    text = text.strip().replace(',', '').upper()

    # Common OCR misreads: 'O'→'0', 'S'→'5', 'I'→'1', 'B'→'8' (only if not suffix), etc.
    corrections = {
        'O': '0',
        'S': '5',
        'I': '1',
        'L': '1',
        'Z': '2',
    }

    # Replace each character using the corrections dictionary, excluding final suffix
    core = text[:-1] if text and text[-1] in "KMB" else text
    suffix = text[-1] if text and text[-1] in "KMB" else ''

    corrected_core = ''.join(corrections.get(c, c) for c in core)

    cleaned_text = corrected_core + suffix

    match = re.match(r'^([0-9]*\.?[0-9]+)\s*([KMB]?)$', cleaned_text)
    if not match:
        return str(0)

    value_str, suffix = match.groups()
    try:
        value = float(value_str)
    except ValueError:
        return str(0)

    multiplier = {
        '': 1,
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'T': 1_000_000_000_000,
    }.get(suffix, 1)

    return str(int(value * multiplier))