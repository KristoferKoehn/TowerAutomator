import ctypes
import threading
import time
from ctypes import wintypes

# Constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

# Map from button name to event flags
BUTTON_DOWN = {
    'left': MOUSEEVENTF_LEFTDOWN,
    'right': MOUSEEVENTF_RIGHTDOWN,
    'middle': MOUSEEVENTF_MIDDLEDOWN
}
BUTTON_UP = {
    'left': MOUSEEVENTF_LEFTUP,
    'right': MOUSEEVENTF_RIGHTUP,
    'middle': MOUSEEVENTF_MIDDLEUP
}

# Structs for ctypes
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.c_ulong),
                ("_input", _INPUT)]

SendInput = ctypes.windll.user32.SendInput
GetSystemMetrics = ctypes.windll.user32.GetSystemMetrics

class MouseController:

    mouse_lock = False
    prev_mouse_position = (0,0)

    def __init__(self):
        self.screen_width = GetSystemMetrics(0)
        self.screen_height = GetSystemMetrics(1)
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)

    def get_mouse_lock(self) -> bool:
        if self.prev_mouse_position != self.get_mouse_lock():
            return True
        else:
            return self.mouse_lock

    def _send_input(self, mi: MOUSEINPUT):
        inp = INPUT(type=INPUT_MOUSE, mi=mi)
        SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _normalize(self, x, y):
        # Convert to 65535-based absolute coordinates
        norm_x = int(x * 65535 / self.screen_width)
        norm_y = int(y * 65535 / self.screen_height)
        return norm_x, norm_y

    def move_to(self, x, y):
        self.mouse_lock = True
        norm_x, norm_y = self._normalize(x, y)
        mi = MOUSEINPUT(dx=norm_x, dy=norm_y,
                        mouseData=0,
                        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                        time=0,
                        dwExtraInfo=None)
        self._send_input(mi)
        self.mouse_lock = False

    def press(self, button='left'):
        mi = MOUSEINPUT(dx=0, dy=0,
                        mouseData=0,
                        dwFlags=BUTTON_DOWN[button],
                        time=0,
                        dwExtraInfo=None)
        self._send_input(mi)

    def release(self, button='left'):
        mi = MOUSEINPUT(dx=0, dy=0,
                        mouseData=0,
                        dwFlags=BUTTON_UP[button],
                        time=0,
                        dwExtraInfo=None)
        self._send_input(mi)

    def click(self, button='left'):
        self.mouse_lock = True
        self.press(button)
        time.sleep(0.01)
        self.release(button)
        self.mouse_lock = False

    def drag(self, start, end, button='left', steps=10, over_time=0.1):
        self.mouse_lock = True
        original_position = self.get_mouse_position()
        self.move_to(*start)
        self.press(button)
        time.sleep(0.05)
        sx, sy = start
        ex, ey = end
        for i in range(1, steps + 1):
            ix = int(sx + (ex - sx) * i / steps)
            iy = int(sy + (ey - sy) * i / steps)
            self.move_to(ix, iy)
            time.sleep(over_time / steps)
        time.sleep(0.5)
        self.release(button)
        self.move_to(*original_position)
        self.mouse_lock = False

    def get_mouse_position(self):
        pt = wintypes.POINT()
        if not self.user32.GetCursorPos(ctypes.byref(pt)):
            raise ctypes.WinError(ctypes.get_last_error())
        return pt.x, pt.y

    def click_return(self, x, y, button = 'left'):
        self.mouse_lock = True
        pos = self.get_mouse_position()
        self.move_to(x, y)
        self.click(button)
        self.move_to(pos[0], pos[1])
        self.mouse_lock = False

    def update_mouse_position(self):
        prev_mouse_pos = self.get_mouse_position()