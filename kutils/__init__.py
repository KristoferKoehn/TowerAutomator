import os
import glob
import platform
import re
import subprocess
import threading
import time
from enum import Enum
import cv2
import pygetwindow as gw


class Menu(Enum):
    ATTACK = "Attack"
    DEFENSE = "Defense"
    UTILITY = "Utility"

def can_image_fit_inside(img_small, img_large):

    h_small, w_small = img_small.shape[:2]
    h_large, w_large = img_large.shape[:2]

    return w_small <= w_large and h_small <= h_large

def find_button_center(image, template, button_name, debug=True, threshold=0.7):
    screen_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not can_image_fit_inside(template, screen_gray):
        print(f"button size wrong, image size is {screen_gray.shape}, template size is {template.shape} on {button_name} ")
        return None

    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        w, h = template.shape[::-1]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2

        if debug:
            def draw(debug_image, x=max_loc[0], y=max_loc[1], _w=w, _h=h):
                cv2.rectangle(debug_image, (x, y), (x + _w, y + _h), (0, 0, 255), 2)
                cv2.putText(debug_image, f"{button_name} {max_val:.3f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            from debug_draw_manager import debug_draw_manager
            debug_draw_manager.set_draw(button_name, draw)
        return center_x, center_y
    else:
        from debug_draw_manager import debug_draw_manager
        debug_draw_manager.clear_debug_draw(button_name)
        return None

def best_button_from_templates(image, templates, threshold=0.650, identifier=""):
    """Detects a button in image using multiple templates. Returns (x, y) or None."""

    # Define region of interest
    height, width = image.shape[:2]
    top = (height * 6) // 20
    bottom = (height * 14) // 20

    roi = image[top:bottom]  # Crop from top 1/8th to 2/3rds

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_score = 0

    for template in templates:
        result = cv2.matchTemplate(gray, template[1], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_match = (max_loc, template[1].shape[::-1])  # ((x, y), (w, h))

    center_x = 0
    center_y = 0
    if best_match and best_score >= 0.4:
        # Adjust coordinates back to full image
        (x, y), (w, h) = best_match
        adjusted_y = y + top  # shift y-coordinate back to original image

        center_x = x + w // 2
        center_y = adjusted_y + h // 2

        # Draw debug info
        c_center = (center_x, center_y)
        c_radius = 40
        c_color = (0, 255, 0)
        c_thickness = 2

        def draw(debug_image):
            cv2.putText(
                debug_image,
                f"{best_score:.3f}",
                (center_x - 30, center_y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                debug_image,
                f"{template[0]}",
                (center_x - 30, center_y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
            cv2.circle(debug_image, c_center, c_radius, c_color, c_thickness)

        from debug_draw_manager import debug_draw_manager
        debug_draw_manager.set_draw(identifier, draw)

    if best_match and best_score >= threshold:
        return center_x, center_y
    return None

def load_templates(template_dir):
    templates = []
    for template_path in glob.glob(os.path.join(template_dir, "*.png")):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"[WARN] Failed to load template: {template_path}")
            continue
        templates.append((template_path, template))
    return templates


def load_image_from_path(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load template: {img_path}")
    return image

def get_game_window(title_contains="The Tower"):
    """Find the Phone Link window."""
    print("Finding Phone Link Window...")
    windows = gw.getWindowsWithTitle(title_contains)
    if len(windows) == 0:
        print("The Tower not found")
        return None
    if windows[0] is not None:
        print("Found Window!")

    return windows[0] if windows else None


def clean_ocr_string(text: str, ignore_list: set[str]) -> str | None:
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).upper()

    if not cleaned or cleaned in ignore_list:
        return None

    return cleaned


def find_executable_by_name(executable_name, directory, recursive=True):
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory: {directory}")

    # Add platform-specific extension if missing (Windows)
    possible_names = [executable_name]
    if platform.system() == "Windows" and not executable_name.lower().endswith(('.exe', '.bat', '.cmd', '.com')):
        possible_names = [f"{executable_name}{ext}" for ext in ['.exe', '.bat', '.cmd', '.com']]

    for root, _, files in os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]:
        for file in files:
            if file in possible_names:
                full_path = os.path.join(root, file)
                if is_executable(full_path):
                    full_path = full_path.replace(".exe", "")
                    return full_path

    return None

def is_executable(path):
    if not os.path.isfile(path):
        return False

    if platform.system() == "Windows":
        return path.lower().endswith(('.exe', '.bat', '.cmd', '.com'))
    else:
        return os.access(path, os.X_OK) and not os.path.isdir(path)

adb_dir = find_executable_by_name("adb.exe", "external_programs")
scrcpy_dir = find_executable_by_name("scrcpy.exe", "external_programs")
display_id = 99999999

def get_id_from_output(line):
    global display_id
    if extract_display_id(line) is not None:
        display_id = extract_display_id(line)
        print(f"Display id=({display_id})")


def handle_output(line):
    if line is not None:
        print(f"-> {line}")

def extract_display_id(log_line: str) -> int | None:
    """Extracts the ID number from a log line like: '[server] INFO: New display: 1440x3120/600 (id=16)'"""
    match = re.search(r'\(id=(\d+)\)', log_line)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'--display-id=(\d+)', log_line)
        if match:
            return int(match.group(1))
    return None

def transform_point_between_resolutions(
    point: tuple[int, int],
    from_resolution: tuple[int, int] = (474,1039),
    to_resolution: tuple[int, int] = (1440,3120)
) -> tuple[int, int]:
    """Transforms a point from one resolution to another, preserving relative position."""
    x, y = point
    from_w, from_h = from_resolution
    to_w, to_h = to_resolution

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    new_x = int(x * scale_x)
    new_y = int(y * scale_y)

    return new_x, new_y

def run_shell_command(program_dir, *args, on_output=handle_output, debug=False):
    command = [program_dir] + list(args)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def read_output(_process=process):
        for line in _process.stdout:
            if on_output:
                on_output(line.rstrip())

    if debug:
        print("[DEBUG] Running shell command: " + " ".join(command))
    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()
    return process  # You can terminate it later if needed

window = get_game_window()

if window is None:
    # ./scrcpy --new-display --no-vd-system-decorations --start-app=com.TechTreeGames.TheTower --window-title="The Tower, Automator" --always-on-top
    test = run_shell_command(
        scrcpy_dir,
        '--new-display',
        '--no-vd-system-decorations',
        '--start-app=com.TechTreeGames.TheTower',
        '--window-title=The Tower, Automator',
        '--always-on-top',
        on_output=get_id_from_output
    )
else:
    print("running on ")
    test = run_shell_command(scrcpy_dir,'--list-displays',on_output=get_id_from_output)

time.sleep(2)
window = get_game_window()


