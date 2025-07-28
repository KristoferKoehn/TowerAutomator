import time
from collections import deque
from collections import Counter
import re


def parse_ocr_number(text: str) -> float | None:
    """Parses a number string like '12.5K', '1M', '100', '4.23B' into a float."""
    text = text.strip().replace(',', '').upper()

    match = re.match(r'^([0-9]*\.?[0-9]+)\s*([KMB]?)$', text)
    if not match:
        return None

    value, suffix = match.groups()
    value = float(value)

    multiplier = {
        '': 1,
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
    }.get(suffix, 1)

    return value * multiplier


class TemporalValueNormalizer:
    def __init__(self, max_age=2.0, max_entries=10):
        """Temporal normalizer for OCR values.

        Args:
            max_age (float): Seconds to retain past values.
            max_entries (int): Max number of values to keep.
        """
        self.history = deque()
        self.max_age = max_age
        self.max_entries = max_entries

    def add_value(self, raw_string):
        """Attempts to parse and add a new OCR string value."""
        try:
            value = float(raw_string)
            now = time.time()
            self.history.append((now, value))
            self._trim_history()
        except ValueError:
            pass  # Ignore unparseable strings

    def get_smoothed_value(self):
        """Returns the smoothed (average) value."""
        self._trim_history()
        if not self.history:
            return 0
        values = [v for _, v in self.history]
        return sum(values) / len(values)

    def get_mode_value(self, rounding=10, min_samples=3) -> float | None:

        if len(self.history) < min_samples:
            return None

        rounded_values = [round(value / rounding) * rounding for _, value in self.history]
        counts = Counter(rounded_values)
        most_common = counts.most_common(1)

        return most_common[0][0] if most_common else None

    def _trim_history(self):
        """Removes old entries from history."""
        now = time.time()
        while self.history and (now - self.history[0][0] > self.max_age or len(self.history) > self.max_entries):
            self.history.popleft()

    def add_money_value(self, raw_string):
        raw_string = raw_string.replace('$', '').strip()
        value = parse_ocr_number(raw_string)
        self.add_value(str(value))