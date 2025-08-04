import os
import threading
import time
import cv2
import numpy as np
import psutil
from PIL import Image

from MouseController import get_mouse_controller
import kutils


def get_menu_color(menu):
    CYAN = (255, 255, 0)
    PINK = (128, 128, 255)
    YELLOW = (0, 255, 255)

    match menu:
        case kutils.Menu.ATTACK:
            return CYAN
        case kutils.Menu.DEFENSE:
            return PINK
        case kutils.Menu.UTILITY:
            return YELLOW
    return None


class DebugDrawManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DebugDrawManager, cls).__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        self._draw_buffer = {}
        self._buffer_lock = threading.Lock()

    def set_draw(self, key, draw_func):
        with self._buffer_lock:
            self._draw_buffer[key] = draw_func

    def clear_debug_draw(self, caller_name: str):
        with self._buffer_lock:
            if caller_name in self._draw_buffer:
                del self._draw_buffer[caller_name]

    def draw_text(self, caller_name: str, lines, start_pos, line_height=15, color=(0, 255, 0)):
        self.clear_contains("_hud_" + caller_name)
        x, y = start_pos
        for i, line in enumerate(lines):
            def draw(debug_image, text=line, pos=(x, y + i * line_height), col=color):
                cv2.putText(debug_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            self.set_draw(f"_hud_{caller_name}_{i}", draw)

    def render_debug_draw(self, image):
        with self._buffer_lock:
            for key, draw_func in self._draw_buffer.items():
                if not key.startswith("_hud"):
                    draw_func(image)

    def render_debug_hud(self, image):
        with self._buffer_lock:
            for key, draw_func in self._draw_buffer.items():
                if key.startswith("_hud"):
                    draw_func(image)

    def clear_contains(self, substr: str):
        with self._buffer_lock:
            keys_to_remove = [k for k in self._draw_buffer if substr in k]
            for k in keys_to_remove:
                del self._draw_buffer[k]

    def clear_unused_debug_draws(self, prefix, keys_this_frame):
        with self._buffer_lock:
            for key in list(self._draw_buffer.keys()):
                if key.startswith(prefix) and key not in keys_this_frame:
                    del self._draw_buffer[key]

    def clear_debug_buffer(self):
        with self._buffer_lock:
            self._draw_buffer.clear()

    def process_debug_image(self, image):

        from debug_logging_manager import get_debug_logger
        dlm = get_debug_logger()

        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)
        GRAY = (128, 128, 128)
        CYAN = (255, 255, 0)
        PINK = (128, 128, 255)
        YELLOW = (0, 255, 255)
        PURPLE = (255, 64, 128)

        debug_draw_manager.render_debug_draw(image)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        debug_output_position = (0, 14)

        width, height = image.size
        crop_top = int(height * 0.04)
        img_cropped = image.crop((0, crop_top, width, height))

        debug_border_width = 350
        new_width = width + debug_border_width
        new_height = height - crop_top

        new_img = Image.new("RGB", (new_width, new_height), color=(0, 0, 0))
        new_img.paste(img_cropped, (debug_border_width, 0))

        # Convert back to OpenCV (NumPy BGR)
        img_np = np.array(new_img)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        pid = os.getpid()
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)

        def line(_text, _color=(0, 255, 0)) -> tuple[str, tuple[int, int, int]]:
            return _text, _color

        input_block = get_mouse_controller().get_mouse_lock()

        lines = [
            line(f"Uptime: {time.time() - dlm.start_time:.2f}", RED),
            line(f"resolution (463x1032): {image.size[0]}x{image.size[1]}"),
            line(f"memory: {rss_mb:.3f} MB"),
            line(""),
            line("=== automation statistics ===", (0, 255, 255)),
        ]

        # Click counts and time since
        for key, (count, time_since) in dlm.click_counts.items():
            if time_since == 0:
                lines.append(line(f"time since {key.lower()}: --"))
            else:
                lines.append(line(f"time since {key.lower()} ({count}): {time.time() - time_since:.2f}"))

        lines.append(line(""))

        lines.append(line("=== current detected resources ===", YELLOW))

        lines.append(line(f"[$] {dlm.resources[0]}"))
        lines.append(
            line(f"(C) {dlm.resources[1]}", YELLOW))
        lines.append(
            line(f"(G) {dlm.resources[2]}", PURPLE))
        lines.append(line(""))

        lines.append(line("=== Upgrade Strategy Data ===", YELLOW))
        t = time.time() - dlm.demon_mode_timer
        lines.append(line(f"current upgrade menu: {dlm.current_upgrade_menu}", get_menu_color(dlm.current_upgrade_menu)))
        lines.append(line(f"cheapest upgrade: {dlm.lowest_cost_upgrade}"))
        lines.append(line(f"Demon mode: {t:.1f}: {dlm.demon_mode_flag}", RED if dlm.demon_mode_flag else GREEN))

        lines.append(line(""))
        lines.append(line("=== visible upgrades ===", YELLOW))

        # Latest result

        for n, v, c, pos in dlm.current_upgrades_detected:
            lines.append(line(str((n, v, c)) + f" {pos[0]}, {pos[1]}"))

        for i in range(5 - len(dlm.current_upgrades_detected)):
            lines.append(line(""))

        lines.append(line("=== total upgrades ===", YELLOW))
        for key, value in dlm.tracked_upgrades.items():
            lines.append(line(f"{key}: ({value[0]}, {value[1]})", get_menu_color(value[2])))

        debug_draw_manager.clear_contains("_hud_debug")
        line_size = 13
        x, y = debug_output_position
        for i, (text, color) in enumerate(lines):
            def draw_line(debug_image, _text=text, _pos=(x, y + i * line_size), col=color):
                cv2.putText(debug_image, _text, _pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

            debug_draw_manager.set_draw(f"_hud_debug_{i}", draw_line)

        debug_draw_manager.render_debug_hud(img_cv2)

        return img_cv2

debug_draw_manager = DebugDrawManager()