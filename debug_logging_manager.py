import sys
import time
import itertools
import threading
import kutils


def make_gradient_spinner(char='●', color_range=range(160, 200)):
    return [f'\033[38;5;{c}m{char}\033[0m' for c in color_range]

class DebugLogger:
    _instance = None
    _lock = threading.Lock()

    resources = ("", "", "") #money, coin, gems
    current_upgrade_menu = kutils.Menu.ATTACK
    current_upgrades_detected = []
    tracked_upgrades = {}
    lowest_cost_upgrade = ""


    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DebugLogger, cls).__new__(cls)
                cls._instance._custom_init()
            return cls._instance

    def _custom_init(self):
        #self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.spinner = itertools.cycle(make_gradient_spinner('◆', range(21, 51)))
        self.last_status_key = None
        self.spinner_active = False
        self.spinner_thread = None
        self._stop_spinner_event = threading.Event()
        #  "Open": (0, 0.0),
        #   "Battle": (0, 0.0),
        self.click_counts = {
            "Claim": (0, 0.0),
            "Retry": (0, 0.0),
            "Sigil": (0, 0.0),

            "Resume": (0, 0.0),
        }
        self.start_time = 0

        self.demon_mode_timer = 0
        self.demon_mode_flag = False

    def log(self, message: str):
        """Print a one-time debug message."""
        print(message)

    def status(self, key: str, message: str):
        """Only print message when key changes."""
        if self.last_status_key != key:
            self.last_status_key = key
            print(message)

    def spinner_status(self, message: str = "Waiting..."):
        """Display a live spinner in-place."""
        if not self.spinner_active:
            self._stop_spinner_event.clear()
            self.spinner_active = True
            self.spinner_thread = threading.Thread(target=self._spinner_loop, args=(message,), daemon=True)
            self.spinner_thread.start()

    def stop_spinner(self, message: str = None):
        """Stop the spinner cleanly."""
        if self.spinner_active:
            self._stop_spinner_event.set()
            self.spinner_thread.join()
            self.spinner_active = False
            sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear line
            sys.stdout.flush()
        if message is not None:
            print(message)


    def _spinner_loop(self, message):
        while not self._stop_spinner_event.is_set():
            spin_char = next(self.spinner)
            sys.stdout.write(f'\r{spin_char} ')  # Only print spinner
            sys.stdout.flush()
            time.sleep(0.1)

    def log_with_spinner_until(self, message: str = "Waiting..."):
        """Always show the spinner after the message; only print the message once."""
        if self.last_status_key != message:
            self.stop_spinner(self.last_status_key)
            self.last_status_key = message
        self._start_spinner_with_message(message)

    def _start_spinner_with_message(self, message: str):
        if not self.spinner_active:
            self._stop_spinner_event.clear()
            self.spinner_active = True
            self.spinner_thread = threading.Thread(
                target=self._spinner_loop_with_message,
                args=(message,),
                daemon=True
            )
            self.spinner_thread.start()

    def _spinner_loop_with_message(self, message: str):
        while not self._stop_spinner_event.is_set():
            spin_char = next(self.spinner)
            sys.stdout.write(f'\r{message} {spin_char}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write(f'{message}')


# Global accessor
def get_debug_logger():
    return DebugLogger()

