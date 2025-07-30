import time

import MouseController
from MouseController import get_mouse_controller
from debug_logging_manager import get_debug_logger
from kutils import load_image_from_path, find_button_center, window
from tracker.strategy_base import ActiveStateStrategy
from PIL import Image


class ResumeStrategy(ActiveStateStrategy):
    """
    Strategy for handling the gameplay screen.
    Runs multiple nested behaviors like gem collection or upgrade scanning.
    """
    resume_match_template = load_image_from_path("ui_templates/resume.png")
    resume_button_template = load_image_from_path("ui_templates/resume.png")

    def __init__(self):
        self.sub_strategies = [
        ]

    def matches(self, screenshot: Image.Image) -> bool:
        marker = find_button_center(screenshot, self.resume_match_template, button_name="gameplay", threshold=0.7)
        return marker

    def on_enter(self):
        get_debug_logger().log_with_spinner_until("[▶️ Resume] Entered Resume state.")
        for strat in self.sub_strategies:
            strat.on_enter()

    def on_exit(self):
        get_debug_logger().log_with_spinner_until("[▶️ Resume] Exiting Resume state.")
        for strat in self.sub_strategies:
            strat.on_exit()

    def run(self, screenshot: Image.Image):
        retry = find_button_center(screenshot, self.resume_button_template, button_name="resume", threshold=0.6)
        if retry:
            abs_x = window.left + retry[0]
            abs_y = window.top + retry[1]
            get_mouse_controller().click_return(abs_x, abs_y)
            get_debug_logger().click_counts["Resume"] = (get_debug_logger().click_counts["Resume"][0] + 1, time.time())
            time.sleep(0.5)
