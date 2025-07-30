from debug_logging_manager import get_debug_logger
from kutils import load_image_from_path, find_button_center
from tracker.strategy_base import ActiveStateStrategy
from strategies.gameplay.gem_collector import GemCollectorStrategy
from strategies.gameplay.upgrade_searcher import UpgradeSearcherStrategy
from PIL import Image


class GameplayStrategy(ActiveStateStrategy):
    """
    Strategy for handling the gameplay screen.
    Runs multiple nested behaviors like gem collection or upgrade scanning.
    """
    gameplay_match_template = load_image_from_path("ui_templates/gameplay_template.png")

    def __init__(self):
        self.sub_strategies = [
            GemCollectorStrategy(),
            UpgradeSearcherStrategy()
        ]

    def matches(self, screenshot: Image.Image) -> bool:
        marker = find_button_center(screenshot, self.gameplay_match_template, button_name="gameplay", threshold=0.4)
        return marker

    def on_enter(self):
        get_debug_logger().log_with_spinner_until("[Gameplay] Entered gameplay state.")
        for strat in self.sub_strategies:
            strat.on_enter()

    def on_exit(self):
        get_debug_logger().log_with_spinner_until("[Gameplay] Exiting gameplay state.")
        for strat in self.sub_strategies:
            strat.on_exit()

    def run(self, screenshot: Image.Image):
        for strat in self.sub_strategies:
            strat.run(screenshot)
