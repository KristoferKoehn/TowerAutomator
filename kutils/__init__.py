import os
import glob
import subprocess
from enum import Enum
import cv2
import pygetwindow as gw
from debug_draw_manager import debug_draw_manager

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
            debug_draw_manager.set_draw(button_name, draw)
        return center_x, center_y
    else:
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
        print("The Tower not found, searching Samsung Flow...")

        windows = gw.getWindowsWithTitle("SM-S928U")
        if len(windows) == 0:
            print("No Samsung Flow window found.")
            return None


        return windows[0]

    if windows[0] is not None:
        print("Found Phone Link Window!")

    return windows[0] if windows else None

def run_adb_shell_command(command):
    try:
        process = subprocess.run(['adb', 'shell', command], capture_output=True, text=True, check=True)
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}\n{e.stderr}"

def run_scrcpy_shell_command(command):
    try:
        process = subprocess.run(['scrcpy', command], capture_output=True, text=True, check=True)
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}\n{e.stderr}"


window = get_game_window()