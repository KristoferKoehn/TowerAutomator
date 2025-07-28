import re
import traceback
import tracemalloc
from datetime import datetime
from enum import Enum
from time import sleep
import os
import psutil
import pygetwindow as gw
from MouseController import MouseController

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print(f"cuda device count: {torch.cuda.device_count()}")
print(f"cuda device name: {torch.cuda.get_device_name(0)}")

import easyocr
import time
import mss
import glob
import numpy as np
from PIL import Image
import cv2  # For template matching
import threading
from pynput import keyboard
from pynput.keyboard import Key



from TemporalValueNormalizer import TemporalValueNormalizer

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
debug_img = None
main_img = None

latest_upgrades_detected = []
latest_upgrades_detected_a = []
latest_upgrades_detected_b = []
current_resources_detected = ("", "", "")

result_lock = threading.Lock()
resource_lock = threading.Lock()

click_counts = {
    "Claim": (0, 0.0),
    "Retry": (0, 0.0),
    "Sigil": (0, 0.0),
    "Open": (0, 0.0),
    "Battle": (0, 0.0),
    "Resume": (0, 0.0),
}

#debug draw functionality start

debug_draw_buffer = {}
debug_draw_lock = threading.Lock()

def set_debug_draw(key, draw_func):
    with debug_draw_lock:
        debug_draw_buffer[key] = draw_func

def clear_debug_draw(caller_name: str):
    """Remove a drawing function."""
    if caller_name in debug_draw_buffer:
        del debug_draw_buffer[caller_name]

def draw_debug_text(caller_name: str, lines, start_pos, line_height=15, color=(0, 255, 0)):
    clear_all_contains("_hud_" + caller_name)
    x, y = start_pos
    for i, line in enumerate(lines):
        def draw(debug_image, text=line, pos=(x, y + i * line_height), col=color):
            cv2.putText(debug_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        set_debug_draw(f"_hud_{caller_name}_{i}", draw)

def render_debug_draw(image):
    with debug_draw_lock:
        for key, draw_func in debug_draw_buffer.items():
            if not key.startswith("_hud"):
                draw_func(image)

def render_debug_hud(image):
    with debug_draw_lock:
        for key, draw_func in debug_draw_buffer.items():
            if key.startswith("_hud"):
                draw_func(image)

def clear_all_contains(substr: str):
    with debug_draw_lock:
        keys_to_remove = [k for k in debug_draw_buffer if substr in k]
        for k in keys_to_remove:
            del debug_draw_buffer[k]

def clear_unused_debug_draws(prefix, keys_this_frame):
    with debug_draw_lock:
        for key in list(debug_draw_buffer.keys()):
            if key.startswith(prefix) and key not in keys_this_frame:
                del debug_draw_buffer[key]

def clear_debug_buffer():
    with debug_draw_lock:
        debug_draw_buffer.clear()
## debug draw end

def upgrade_detection_worker(left):
    global latest_upgrades_detected_a, latest_upgrades_detected_b

    templates = load_templates("upgrade_template")
    while True:
        try:
            if main_img is not None:
                result = detect_upgrade_templates(main_img, templates, left)

                # Safely update global variable
                with result_lock:
                    if left:
                        latest_upgrades_detected_a = result
                    else:
                        latest_upgrades_detected_b = result
        except Exception as e:
            with open("thread_errors.log", "a") as f:
                f.write("Exception in upgrade_detection_worker:\n")
                traceback.print_exc(file=f)
            raise  # Re-raise to allow the thread to crash loudly if desired


def resource_detection_worker():
    global current_resources_detected
    while True:
        try:
            if main_img is not None:
                result = resource_info(main_img)

                # Safely update global variable
                with resource_lock:
                    current_resources_detected = result
        except Exception as e:
            with open("thread_errors.log", "a") as f:
                f.write("Exception in resource_detection_worker:\n")
                traceback.print_exc(file=f)
            raise

def memory_logger(interval_seconds=30, trace_top=5):
    pid = os.getpid()
    process = psutil.Process(pid)

    tracemalloc.start()

    while True:
        try:
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size (MB)

            log_lines = [f"{datetime.now().isoformat()} | Memory RSS: {rss_mb:.2f} MB"]

            # Take and analyze memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            log_lines.append("[Top memory allocations:]")
            for i, stat in enumerate(top_stats[:trace_top]):
                log_lines.append(f"  {i+1}: {stat}")

            log_lines.append("\n")

            with open("memory_usage.log", "a") as f:
                f.write("\n".join(log_lines))

        except Exception as e:
            with open("thread_errors.log", "a") as f:
                f.write("Exception in memory_logger:\n")
                import traceback
                traceback.print_exc(file=f)

        sleep(interval_seconds)

def load_templates(template_dir):
    templates = []
    for template_path in glob.glob(os.path.join(template_dir, "*.png")):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"[WARN] Failed to load template: {template_path}")
            continue
        templates.append((template_path, template))
    return templates

def load_image_from_path(img_dir):
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load template: {img_dir}")
    return image

## start input block detection logic

key_pressed = set()
input_lock = threading.Lock()

#track screenshot keys
_screenshot_count = 0
_up_pressed = False

mouse_controller = MouseController()

def on_press(key):
    global _screenshot_count, _up_pressed
    with input_lock:
        if key == Key.down:
            if key in key_pressed:
                key_pressed.discard(key)
            else:
                key_pressed.add(key)
        if key == keyboard.Key.up and not _up_pressed:
            _up_pressed = True
            # Simulate an image to save
            image = main_img.copy()
            os.makedirs("screenshots", exist_ok=True)
            filename = f"screenshots/screenshot_{_screenshot_count}.png"
            cv2.imwrite(filename, image)
            print(f"[✓] Screenshot saved to {filename}")
            _screenshot_count += 1

def on_release(key):
    global _up_pressed
    if key == keyboard.Key.up:
        _up_pressed = False

    #with input_lock:
        #key_pressed.discard(key)

# Start listeners
keyboard.Listener(on_press=on_press, on_release=on_release).start()

def any_input_held():
    with input_lock:
        return bool(key_pressed)

## end input block detection logic

def detect_sigil_from_templates(image, templates, threshold=0.650):
    """Detects sigil in image using multiple rotated templates. Returns (x, y) or None."""

    # Define region of interest
    height, width = image.shape[:2]
    top = (height * 6) // 20
    bottom = (height * 14) // 20

    roi = image[top:bottom]  # Crop from top 1/8th to 2/3rds

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_score = 0

    for template in templates:
        result = cv2.matchTemplate(gray, template[1], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_match = (max_loc, template[1].shape[::-1])  # ((x, y), (w, h))

    center_x = 0
    center_y = 0
    if best_match and best_score >= 0.4:
        # Adjust coordinates back to full image
        (x, y), (w, h) = best_match
        adjusted_y = y + top  # shift y-coordinate back to original image

        center_x = x + w // 2
        center_y = adjusted_y + h // 2

        # Draw debug info
        c_center = (center_x, center_y)
        c_radius = 40
        c_color = (0, 255, 0)
        c_thickness = 2

        def draw(debug_image):
            cv2.putText(
                debug_image,
                f"{best_score:.3f}",
                (center_x - 30, center_y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
            cv2.circle(debug_image, c_center, c_radius, c_color, c_thickness)

        set_debug_draw("sigil", draw)

    if best_match and best_score >= threshold:
        print(f"Best match: {center_x},{center_y} with score {best_score:.3f}")
        return center_x, center_y
    return None

def get_phone_window(title_contains="The Tower"):
    """Find the Phone Link window."""
    windows = gw.getWindowsWithTitle(title_contains)
    return windows[0] if windows else None


def find_button_center(image, template, button_name, debug=True, threshold=0.7):
    screen_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        w, h = template.shape[::-1]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2

        if debug:
            def draw(debug_image, x=max_loc[0], y=max_loc[1], _w=w, _h=h):
                cv2.rectangle(debug_image, (x, y), (x + _w, y + _h), (0, 0, 255), 2)
                cv2.putText(debug_image, button_name, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            set_debug_draw(button_name, draw)
        return center_x, center_y
    else:
        clear_debug_draw(button_name)
        return None

def capture_window(win):
    """Capture a screenshot of the given window using mss (multi-monitor compatible)."""
    bbox = {
        "top": win.top,
        "left": win.left,
        "width": win.width,
        "height": win.height
    }

    with mss.mss() as sct:
        sct_img = sct.grab(bbox)
        image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return np.array(image)


def resource_info(image):
    crop_x, crop_y = 24, 50  # top-left corner of the crop
    crop_w, crop_h = 130, 160

    result = ("--", "--", "--")
    def draw(_image):
        cv2.rectangle(_image, (crop_x, crop_y), (crop_w, crop_h), (255, 0, 0), 2)
    set_debug_draw("money", draw)

    top_left_crop = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    ocr_result = reader.readtext(top_left_crop)

    # Sort OCR results by vertical position (top to bottom)
    ocr_result_sorted = sorted(ocr_result, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)

    if len(ocr_result_sorted) >= 3:
        label_tuple = tuple(text.strip() for (_, text, _) in ocr_result_sorted[:3])
        result = label_tuple

    return result

def detect_upgrade_templates(image, templates, left=True, threshold=0.65):
    if image is None:
        raise ValueError("Input image is None")

    results = []
    debug_keys_this_frame = []

    height = image.shape[0]
    width = image.shape[1]
    lower_third_start = int(height * 2 / 3)
    w_offset = 0

    if left:
        lower_third = image[lower_third_start:, :width // 2]
    else:
        w_offset = width // 2
        lower_third = image[lower_third_start:, width // 2:]

    search_sector = cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY)
    boxes = []
    confidences = []

    for template in templates:

        res = cv2.matchTemplate(search_sector, template[1], cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            full_y = pt[1] + lower_third_start
            box = [w_offset + pt[0], full_y, 200, 80]  # x, y, w, h
            conf = float(res[pt[1], pt[0]])
            boxes.append(box)
            confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=0.6)

    if len(indices) > 0:
        for box_index, i in enumerate(indices.flatten()):
            x, y, w, h = boxes[i]
            conf = confidences[i]

            color = (255, 255, 0)
            roi = image[y:y + h, x:x + w]
            ocr_result = reader.readtext(roi)

            upgrade_name = ""
            value = ""
            cost = ""

            for text_index, (bbox, text, conf) in enumerate(ocr_result):
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

                key_label = f"upgrade{left}_label_{box_index}_{text_index}"
                set_debug_draw(key_label, draw_label)
                debug_keys_this_frame.append(key_label)

            results.append((upgrade_name.strip(), value.strip(), cost.strip(), (x + 165, y + 40)))
            def draw_box(debug_image, _x=x, _y=y, _w=w, _h=h, _color=color, _conf=conf, _upgrade_name=upgrade_name.strip()):
                cv2.rectangle(debug_image, (_x, _y), (_x + _w, _y + _h), _color, 2)
                cv2.putText(debug_image, f"{_upgrade_name}, {_conf:.2f}", (_x + 5, _y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.circle(debug_image, (_x + 165, _y + 40), 4, _color, 2)

            key_box = f"upgrade{left}_box_{box_index}"
            set_debug_draw(key_box, draw_box)
            debug_keys_this_frame.append(key_box)

    clear_unused_debug_draws("upgrade{left}", debug_keys_this_frame)
    return results


def process_debug_image(image):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    CYAN = (255, 255, 0)
    PINK = (128, 128, 255)
    YELLOW = (0, 255, 255)
    PURPLE = (255, 64, 128)

    render_debug_draw(image)

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    debug_output_position = (0, 20)

    width, height = image.size
    crop_top = int(height * 0.04)
    img_cropped = image.crop((0, crop_top, width, height))

    debug_border_width = 300
    new_width = width + debug_border_width
    new_height = height - crop_top

    new_img = Image.new("RGB", (new_width, new_height), color=(0, 0, 0))
    new_img.paste(img_cropped, (debug_border_width, 0))

    # Convert back to OpenCV (NumPy BGR)
    img_np = np.array(new_img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)

    def line(_text, _color=(0, 255, 0)) -> tuple[str, tuple[int, int, int]]:
        return _text, _color

    lines = [
        line(f"input block: {any_input_held()}", RED if any_input_held() else GREEN),
        line(f"Uptime: {time.time() - start_time:.2f}", RED),
        line(f"resolution (463x1032): {image.size[0]}x{image.size[1]}"),
        line(f"memory: {rss_mb:.3f} MB"),
        line(""),
        line("=== automation statistics ===", (0, 255, 255)),
    ]

    # Click counts and time since
    for key, (count, time_since) in click_counts.items():
        if time_since == 0:
            lines.append(line(f"time since {key.lower()}: --"))
        else:
            lines.append(line(f"time since {key.lower()} ({count}): {time.time() - time_since:.2f}"))

    lines.append(line(""))

    lines.append(line("=== current detected resources ===", YELLOW))
    with resource_lock:
        lines.append(line(f"[$] {cash_value_normalizer.get_mode_value()} | {current_resources_detected[0]}"))
        lines.append(line(f"(C) {coin_value_normalizer.get_mode_value()} | {current_resources_detected[1]}", YELLOW))
        lines.append(line(f"(G) {int(gem_value_normalizer.get_smoothed_value())} | {current_resources_detected[2]}", PURPLE))
    lines.append(line(""))

    lines.append(line("=== Upgrade Strategy Data ===", YELLOW))
    m = (0,0,0)
    match current_menu:
        case Menu.ATTACK:
            m = CYAN
        case Menu.DEFENSE:
            m = PINK
        case Menu.UTILITY:
            m = YELLOW

    lines.append(line(f"current menu: {current_menu}", m))
    lines.append(line(f"cheapest upgrade: {lowest_known_cost_item}"))

    lines.append(line(""))
    lines.append(line("=== visible upgrades ===", YELLOW))

    # Latest result
    with result_lock:
        for n, v, c, pos in latest_upgrades_detected:
            lines.append(line(str((n, v, c)) + f" {pos[0]}, {pos[1]}"))

        for i in range(5 - len(latest_upgrades_detected)):
            lines.append(line(""))

    lines.append(line("=== total upgrades ===", YELLOW))
    for key, value in tracked_upgrades.items():
        lines.append(line(f"{key}: {value}"))

    clear_all_contains("_hud_debug")
    line_size = 14
    x, y = debug_output_position
    for i, (text, color) in enumerate(lines):
        def draw_line(debug_image, _text=text, _pos=(x, y + i * line_size), col=color):
            cv2.putText(debug_image, _text, _pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        set_debug_draw(f"_hud_debug_{i}", draw_line)

    render_debug_hud(img_cv2)

    return img_cv2


tracked_upgrades = {}

cash_value_normalizer = TemporalValueNormalizer(max_age=5.0)
coin_value_normalizer = TemporalValueNormalizer(max_age=5.0)
gem_value_normalizer = TemporalValueNormalizer(max_age=5.0)

rejected_names = ["RapidChance(Fire", "DamageThorn",
                  "LandDamageMine", "LandChanceMine",
                  "LandNineChance", "LandnineChance",
                  "RapidDurationFire", "RapidChanceFire"
                  "LandRadiusMine", "AttackUpgradeFree",
                  "LandNineRadius", "LandNineDamage",
                  "naxRecovery", "CoinsKcillBonus",
                  "CoinsJave", "Defense00", "DefenseOO"
                  "InterestJave", "cashBonus", "cashlave",
                  "cashlave", "RecoveryMax", "CoinsBonusKill",
                  "ShockiaveSize", "BounceShotchance",
                  "AttackUpgradeFree=", "Packagechance"
                  "nultishotChance", "nultishotTargets",
                  "nhultishotTargets", "BounceShot;Targets"
                  "RapidChanceFire:", "Rapid[ChanceFire",
                  "Aosolute", "Kegen"
                  ]

def update_visible_upgrades():
    global tracked_upgrades, latest_upgrades_detected, latest_upgrades_detected_a, latest_upgrades_detected_b
    with resource_lock:
        latest_upgrades_detected.clear()
        latest_upgrades_detected.extend(latest_upgrades_detected_a)
        latest_upgrades_detected.extend(latest_upgrades_detected_b)

        for name, value, cost, position in latest_upgrades_detected:
            if len(name) > 0:
                n = name.replace(' ', '').replace('/', '').replace(':', '').replace('[', '')
                if n not in rejected_names:
                    tracked_upgrades[n] = (value, normalize_cost_value(cost), current_menu)

def clear_visible_upgrades():
    with resource_lock:
        latest_upgrades_detected.clear()

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

lowest_known_cost_item = ""

def retry_cleanup():
    tracked_upgrades.clear()

class Menu(Enum):
    ATTACK = "Attack"
    DEFENSE = "Defense"
    UTILITY = "Utility"

current_menu = Menu.ATTACK

# === Main Loop ===
if __name__ == "__main__":

    #left_thread = threading.Thread(target=upgrade_detection_worker, args=[True], daemon=True)
    #left_thread.start()

    #right_thread = threading.Thread(target=upgrade_detection_worker, args=[False], daemon=True)
    #right_thread.start()

    thread = threading.Thread(target=resource_detection_worker, daemon=True)
    thread.start()

    threading.Thread(target=memory_logger, daemon=True).start()
    window = get_phone_window()

    if not window:
        print("[WARN] Could not find window.")
        exit()

    print("Starting main loop...")

    start_time = time.time()
    time_since_sigil = 0
    time_since_claim = 0
    time_since_retry = 0

    sigil_templates = load_templates("sigil_templates") # this is a list of images

    retry_template = load_image_from_path("retry_template/retry.png")
    open_template = load_image_from_path("ui_templates/Open.png")
    claim_template = load_image_from_path("claim_template/claim.png")
    resume_template = load_image_from_path("ui_templates/resume.png")
    battle_template = load_image_from_path("ui_templates/battle.png")
    attack_upgrade_template = load_image_from_path("ui_templates/attack_upgrades.png")
    defense_upgrade_template = load_image_from_path("ui_templates/defense_upgrades.png")
    utility_upgrade_template = load_image_from_path("ui_templates/utility_upgrades.png")
    attack_button_template = load_image_from_path("ui_templates/attack_menu_button.png")
    defense_button_template = load_image_from_path("ui_templates/defense_menu_button.png")
    utility_button_template = load_image_from_path("ui_templates/utility_menu_button.png")

    def switch_menus():
        match current_menu:
            case Menu.ATTACK:
                menu = find_button_center(main_img, defense_button_template,"defense_menu")
                if menu:
                    abs_x = window.left + menu[0]
                    abs_y = window.top + menu[1]
                    mouse_controller.click_return(abs_x, abs_y)
            case Menu.DEFENSE:
                menu = find_button_center(main_img, utility_button_template, "utility_menu")
                if menu:
                    abs_x = window.left + menu[0]
                    abs_y = window.top + menu[1]
                    mouse_controller.click_return(abs_x, abs_y)
            case Menu.UTILITY:
                menu = find_button_center(main_img, attack_button_template, "attack_menu")
                if menu:
                    abs_x = window.left + menu[0]
                    abs_y = window.top + menu[1]
                    mouse_controller.click_return(abs_x, abs_y)


    scroll_direction_flag = True  # ensure this is defined globally


    def scroll_menu():
        global scroll_direction_flag, main_img, window
        height, width = main_img.shape[:2]

        # Calculate start/end positions
        start_y = int(height * (0.75 if scroll_direction_flag else 0.85)) + window.top
        start_x = width // 2 + window.left
        direction = 1 if scroll_direction_flag else -1
        drag_distance = int(window.height * 0.085) # two menu objects are about 17% of the screen
        end_y = start_y + direction * drag_distance

        mouse_controller.drag((start_x, start_y), (start_x, end_y))

        scroll_direction_flag = not scroll_direction_flag


    menu_scroll_timer = time.time()
    menu_change_timer = time.time()

    menu_upgrade_lockout_timer = time.time() + 2.0

    frame_count = 0
    while True:
        img = capture_window(window)
        debug_img = img.copy()
        main_img = img
        matched = False

        with resource_lock:
            cash_value_normalizer.add_money_value(current_resources_detected[0])
            coin_value_normalizer.add_money_value(current_resources_detected[1])
            gem_value_normalizer.add_value(current_resources_detected[2])


        lowest_cost = float("inf")
        for k, v in tracked_upgrades.items():
            cost_val = v[1]

            if cost_val is None or cost_val == "MAX":
                continue

            try:
                cost_float = float(cost_val)

                if cost_float < lowest_cost:
                    lowest_cost = cost_float
                    lowest_known_cost_item = k

            except (ValueError, TypeError):
                print(f"Invalid cost value for {k}: {cost_val}")


        if not any_input_held() or not mouse_controller.get_mouse_lock():

            claim_pos = find_button_center(img, claim_template,"claim_template/claim.png") if frame_count % 5 == 1 else None
            if claim_pos:
                abs_x = window.left + claim_pos[0]
                abs_y = window.top + claim_pos[1]
                mouse_controller.click_return(abs_x, abs_y)
                click_counts["Claim"] = (click_counts["Claim"][0] + 1, time.time())

            retry_pos = find_button_center(img, retry_template,"retry_template/retry.png") if frame_count % 5 == 0 else None
            if retry_pos:
                abs_x = window.left + retry_pos[0]
                abs_y = window.top + retry_pos[1]
                mouse_controller.click_return(abs_x, abs_y)
                click_counts["Retry"] = (click_counts["Retry"][0] + 1, time.time())
                retry_cleanup()

            resume_pos = find_button_center(img, resume_template,"ui_templates/resume.png") if frame_count % 10 == 0 else None
            if resume_pos:
                abs_x = window.left + resume_pos[0]
                abs_y = window.top + resume_pos[1]
                mouse_controller.click_return(abs_x, abs_y)
                click_counts["Resume"] = (click_counts["Resume"][0] + 1, time.time())
            else:
                battle_pos = find_button_center(img, battle_template,"ui_templates/battle.png") if frame_count % 50 == 0 else None
                if battle_pos:
                    abs_x = window.left + battle_pos[0]
                    abs_y = window.top + battle_pos[1]
                    mouse_controller.click_return(abs_x, abs_y)
                    click_counts["Battle"] = (click_counts["Battle"][0] + 1, time.time())

            open_pos = find_button_center(img, open_template, "ui_templates/Open.png") if frame_count % 25 == 6 else None
            if open_pos:
                abs_x = window.left + open_pos[0]
                abs_y = window.top + open_pos[1]
                mouse_controller.click_return(abs_x, abs_y)
                sleep(4)
                click_counts["Open"] = (click_counts["Open"][0] + 1, time.time())

            sigil_pos = detect_sigil_from_templates(img, sigil_templates) if frame_count % 2 == 0 else None
            if sigil_pos:
                abs_x = window.left + sigil_pos[0]
                abs_y = window.top + sigil_pos[1]
                mouse_controller.click_return(abs_x, abs_y)
                click_counts["Sigil"] = (click_counts["Sigil"][0] + 1, time.time())

            if time.time() - menu_change_timer > 30:
                menu_change_timer = time.time()
                #switch_menus()

            if time.time() - menu_scroll_timer > 5:
                menu_scroll_timer = time.time()
                #scroll_menu()

        # end mouse lockout. Don't move the mouse around outside of that block 👆


        prev_menu = current_menu
        #detect menu state
        attack_menu = find_button_center(img, attack_upgrade_template, "attack menu", threshold=0.9)
        if attack_menu:
            current_menu = Menu.ATTACK
        defense_menu = find_button_center(img, defense_upgrade_template, "defense menu", threshold=0.9)
        if defense_menu:
            current_menu = Menu.DEFENSE
        utility_menu = find_button_center(img, utility_upgrade_template, "utility menu", threshold=0.9)
        if utility_menu:
            current_menu = Menu.UTILITY



        if prev_menu != current_menu:
            menu_upgrade_lockout_timer = time.time()

        if time.time() - menu_upgrade_lockout_timer > 1:
            update_visible_upgrades()
        else:
            clear_visible_upgrades()



        #end of frame cleanup

        mouse_controller.update_mouse_position()

        processed_img = process_debug_image(debug_img)
        cv2.imshow("The Tower Automator", processed_img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to break loop
            break

        frame_count += 1
        if frame_count % 1000 == 0:
            frame_count = 0
            clear_debug_buffer()


    '''
    get going on the scrolling and menu traversal. just gotta do up/down/up then switch to next guy.
    
    if lowest thing spotted, then click and go on cooldown (less than a second, probably like 0.2s)
    
    start thinking about prioritization. things to max early, things to stick on. nearly there
    
    there's probably a bunch of bugs lurking under the hood. going into night 3 with 400s uptime.
    8.88m coins
    1222 gems
    '''