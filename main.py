import re
import traceback
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from time import sleep
import os
import psutil

from debug_logging_manager import get_debug_logger
from kutils import find_button_center, load_image_from_path, load_templates, window, run_adb_shell_command, \
    run_scrcpy_shell_command
from ocr_worker import ocr_pool
from MouseController import MouseController, get_mouse_controller
from debug_draw_manager import debug_draw_manager

import time
import mss
import numpy as np
from PIL import Image
import cv2  # For template matching
import threading
from pynput import keyboard
from pynput.keyboard import Key

from TemporalValueNormalizer import TemporalValueNormalizer
from strategies.gameplay.gameplay import GameplayStrategy
from strategies.reopen.reopen import ReOpenStrategy
from strategies.resume.resume import ResumeStrategy
from strategies.retry.retry import RetryStrategy
from tracker.menu_tracker import MenuTracker

debug_img = None
main_img = None

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

## start input block detection logic

key_pressed = set()
input_lock = threading.Lock()

#track screenshot keys
_screenshot_count = 0
_up_pressed = False

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
            print(f"[📸] Screenshot saved to {filename}")
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

# === Main Loop ===
if __name__ == "__main__":

    get_debug_logger().log_with_spinner_until("resource detection thread started")
    threading.Thread(target=memory_logger, daemon=True).start()
    get_debug_logger().log_with_spinner_until("memory_logger thread started")

    run_scrcpy_shell_command("--new-display --no-vd-system-decorations --start-app=com.TechTreeGames.TheTower")


    if not window:
        print("[WARN] Could not find window.")
        exit()

    get_debug_logger().log_with_spinner_until("Starting main loop...")

    get_debug_logger().start_time = time.time()
    time_since_sigil = 0
    time_since_claim = 0
    time_since_retry = 0

    resume_template = load_image_from_path("ui_templates/resume.png")
    battle_template = load_image_from_path("ui_templates/battle.png")

    attack_button_template = load_image_from_path("ui_templates/attack_menu_button.png")
    defense_button_template = load_image_from_path("ui_templates/defense_menu_button.png")
    utility_button_template = load_image_from_path("ui_templates/utility_menu_button.png")

    def scroll_menu():
        height, width = main_img.shape[:2]

        # Calculate start/end positions
        start_y = int(height * (0.85)) + window.top
        start_x = width // 2 + window.left
        drag_distance = int(window.height * 0.085) # two menu objects are about 17% of the screen
        end_y = start_y - drag_distance

        get_mouse_controller().drag((start_x, start_y), (start_x, end_y))


    menu_scroll_timer = time.time()
    menu_change_timer = time.time()

    menu_upgrade_lockout_timer = time.time() + 2.0

    frame_count = 0

    menu_tracker = MenuTracker(
        [
            GameplayStrategy(),
            RetryStrategy(),
            ReOpenStrategy(),
            ResumeStrategy(),
        ]
    )

    while True:
        img = capture_window(window)
        debug_img = img.copy()
        main_img = img
        matched = False
        menu_tracker.update(main_img)


        if not any_input_held() and not get_mouse_controller().get_mouse_lock():

            resume_pos = find_button_center(img, resume_template,"ui_templates/resume.png") if frame_count % 10 == 0 else None
            if resume_pos:
                abs_x = window.left + resume_pos[0]
                abs_y = window.top + resume_pos[1]
                #get_mouse_controller().click_return(abs_x, abs_y)
                #click_counts["Resume"] = (click_counts["Resume"][0] + 1, time.time())
            else:
                battle_pos = find_button_center(img, battle_template,"ui_templates/battle.png") if frame_count % 50 == 0 else None
                if battle_pos:
                    abs_x = window.left + battle_pos[0]
                    abs_y = window.top + battle_pos[1]
                    #get_mouse_controller().click_return(abs_x, abs_y)
                    #click_counts["Battle"] = (click_counts["Battle"][0] + 1, time.time())

            if time.time() - menu_scroll_timer > 5:
                menu_scroll_timer = time.time()
                #scroll_menu()

        # end mouse lockout. Don't move the mouse around outside of that block 👆



        #end of frame cleanup

        get_mouse_controller().update_mouse_position()

        processed_img = debug_draw_manager.process_debug_image(debug_img)
        cv2.imshow("The Tower Automator", processed_img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to break loop
            break

        frame_count += 1
        if frame_count % 1000 == 0:
            frame_count = 0
            debug_draw_manager.clear_debug_buffer()

'''
7/30/2025
Accomplishments:
-made debug log manager
-cleaned up main/ moved all upgrade search stuff to upgrade_searcher.py
-got things to run. leveraged near 100% of old code, just rearranged into better architecture
-made upgrade detection *waaay* faster
-pulled all ocr out of functions, now have ocr pool to request.
    -upgrade OCRs on slices which are OCR'd in parallel all at once, real fast.

TODOS:
    -get menu shmovement working
        -gather stage (just scroll down, flick up, then next menu) to get all the costs
        -hunt stage (go to menu of target upgrade, scroll down incrementally until found, hammer click)
    -hook up the rest of debuglogging in upgrade_searcher.
        -hook up current state in debuglogging as well. gotta get that sick shit going
            -ascii animations?? hell yeah.
    
    -start thinking about other states. card modify? claim missions/events? probably. I want this shit to be like 10 
minutes a day tops. hammer through card packs, upgrades, then go about my day. I want to wake up at wave 4500+

    -need to capture wave number for "max wave reached" statistic.
    -think about "starting gem/coin count" implementation for "x gems, y coins farmed" stats
    
    -there's a somewhat infrequent disconnect issue. trying wired for better connection?? or at least plugged in/charging
    -sometimes on quit there's a "rate this app" dialogue that will one-shot the automator if I don't catch.

    -implement a better hands-off toggle. gotta press a button or do like an easy software button or something to turn
off having the mouse yanked randomly.

let's see if it survives the night:  
25.69M coins
1307 gems
success : TBD

7/29/2025
TODOS:
get gem collector finished->
    sigil and claim are ready to go. just need to hook up the toggle and delete shit from main and initialize

get upgrade guy going->
    we get a great environment for this now. don't care about maintaining state, we can scout fast, like <45s
    - need to get OCR data cleaning really tight and with fallbacks. don't want to be stuck trying to upgrade 'rorbs'
    upgrade seeking will be awesome.
        -go to menu that has upgrade
        -go down the list until we see it
        -click on it.
        -when leaving, scroll to the top. this might be psycho fast? like quick flick
        

think about debug system. Gem collector needs to say "hey I clicked on something at this time" and debugger needs to say
"it's been 0 seconds since this happened" and count up like it does now.

mouse controller having the physical input lock toggle would be great, maybe switch over to ctrl-hold for now?
'''