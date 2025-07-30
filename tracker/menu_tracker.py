import itertools
import sys
import time
from typing import Optional

from debug_logging_manager import get_debug_logger
from tracker.strategy_base import DetectionStrategy, ActiveStateStrategy
from PIL import Image  # or np.ndarray, depending on your screenshot format


class MenuTracker:
    """
    Tracks which screen/menu is currently active, and runs associated logic.
    """

    def __init__(self, strategies: list[ActiveStateStrategy]):
        self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.strategies = strategies
        self.current_strategy: Optional[ActiveStateStrategy] = None


    frame_count = 0
    def update(self, screenshot: Image.Image):
        """
        Call this once per frame with the current game screenshot.
        Will detect and update current strategy, handling state transitions and logic execution.
        """

        matched_strategy = self._detect_state(screenshot)

        # Handle state transition
        if matched_strategy is not self.current_strategy:
            if self.current_strategy:
                self._exit_state(self.current_strategy)
            if matched_strategy:
                self._enter_state(matched_strategy)
            self.current_strategy = matched_strategy

        # Run active strategy if present
        if self.current_strategy:
            self.current_strategy.run(screenshot)
        else:
            get_debug_logger().log_with_spinner_until("No current strategy. hilarious!")


        self.frame_count += 1
        if self.frame_count > 300 == 0:
            self.frame_count = 0

    def _detect_state(self, screenshot: Image.Image) -> Optional[ActiveStateStrategy]:
        """
        Checks each strategy in order to determine the current screen.
        The first one to return True from matches() is selected.
        """
        for strategy in self.strategies:
            if strategy.matches(screenshot):
                return strategy
        return None

    def _enter_state(self, strategy: ActiveStateStrategy):
        if hasattr(strategy, "on_enter"):
            strategy.on_enter()

    def _exit_state(self, strategy: ActiveStateStrategy):
        if hasattr(strategy, "on_exit"):
            strategy.on_exit()


    def show_idle_spinner(self, msg="No current strategy", delay=0.2):
        sys.stdout.write(f'\r{msg} {next(self.spinner)}')
        sys.stdout.flush()
        time.sleep(delay)