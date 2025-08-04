import os
import time
from datetime import datetime

import cv2

import MouseController
from MouseController import get_mouse_controller
from debug_logging_manager import get_debug_logger
from kutils import load_image_from_path, find_button_center, window
from tracker.strategy_base import ActiveStateStrategy
from PIL import Image


class RetryStrategy(ActiveStateStrategy):
    """
    Strategy for handling the gameplay screen.
    Runs multiple nested behaviors like gem collection or upgrade scanning.
    """
    retry_template = load_image_from_path("ui_templates/retry_template.png")
    retry_button_template = load_image_from_path("ui_templates/retry.png")

    def __init__(self):
        self.sub_strategies = [
        ]

    def matches(self, screenshot: Image.Image) -> bool:
        marker = find_button_center(screenshot, self.retry_template, button_name="gameplay", threshold=0.7)
        return marker

    def on_enter(self):
        get_debug_logger().log_with_spinner_until("[🔂 Retry] Entered Retry state.")
        for strat in self.sub_strategies:
            strat.on_enter()

    def on_exit(self):
        get_debug_logger().log_with_spinner_until("[🔂 Retry] Exiting Retry state.")
        for strat in self.sub_strategies:
            strat.on_exit()

    def run(self, screenshot: Image.Image):
        os.makedirs("screenshots", exist_ok=True)
        filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, screenshot)
        retry = find_button_center(screenshot, self.retry_button_template, button_name="gameplay", threshold=0.6)
        if retry:
            MouseController.touch_position(retry[0], retry[1])
            get_debug_logger().click_counts["Retry"] = (get_debug_logger().click_counts["Retry"][0] + 1, time.time())
            time.sleep(0.5)
