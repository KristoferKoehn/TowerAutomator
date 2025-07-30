from abc import ABC, abstractmethod

class DetectionStrategy(ABC):
    """
    Base interface for all screen state detection strategies.
    Used to determine if the game is in a specific menu or state.
    """

    @abstractmethod
    def matches(self, screenshot) -> bool:
        """
        Return True if this strategy's screen state is currently active.
        """
        pass


class ActiveStateStrategy(DetectionStrategy):
    """
    Extended strategy for states that require active logic (e.g., automation behavior).
    Called when this state is detected as active.
    """

    @abstractmethod
    def run(self, screenshot) -> None:
        """
        Execute automation logic for the active state.
        Only called when matches() has returned True.
        """
        pass


class SubStrategy(ABC):
    """
    Sub-behavior strategy for a more complex parent state (like gameplay).
    Used to break down gameplay into modular tasks (combat, farm, heal, etc.).
    """

    @abstractmethod
    def matches(self, screenshot) -> bool:
        """
        Determines if this sub-strategy should run on the current frame.
        """
        pass

    @abstractmethod
    def run(self, screenshot) -> None:
        """
        Perform the automation logic for this sub-strategy.
        """
        pass
