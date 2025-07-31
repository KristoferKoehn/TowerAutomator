import time

from debug_logging_manager import get_debug_logger
from kutils import load_templates, load_image_from_path, find_button_center, window, best_button_from_templates
from tracker.strategy_base import SubStrategy
from PIL import Image
from MouseController import touch_position


class GemCollectorStrategy(SubStrategy):
    """
    Looks for gem popups and clicks them.
    """
    sigil_templates = load_templates("sigil_templates")
    claim_template = load_image_from_path("ui_templates/claim.png")


    def __init__(self):
        pass

    def matches(self, screenshot) -> bool:
        pass

    def on_enter(self):
        get_debug_logger().log_with_spinner_until("[💎 GemCollector] Starting gem collection behavior.")

    def on_exit(self):
        get_debug_logger().log_with_spinner_until("[💎 GemCollector] Cleaning up.")

    frame_count = 0
    def run(self, screenshot: Image.Image):
        claim_pos = find_button_center(screenshot, self.claim_template, "claim_template/claim.png") if self.frame_count % 4 == 0 else None
        if claim_pos:
            touch_position(claim_pos[0], claim_pos[1])
            get_debug_logger().click_counts["Claim"] = (get_debug_logger().click_counts["Claim"][0] + 1, time.time())
        sigil_pos = best_button_from_templates(screenshot, self.sigil_templates, identifier="sigil_templates") if self.frame_count % 4 == 2 else None
        if sigil_pos:
            touch_position(sigil_pos[0], sigil_pos[1])
            get_debug_logger().click_counts["Sigil"] = (get_debug_logger().click_counts["Sigil"][0] + 1, time.time())
        self.frame_count = self.frame_count + 1
        if self.frame_count == 400:
            self.frame_count = 0