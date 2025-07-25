import pygetwindow as gw
import pyautogui
import easyocr
import time
import mss
import glob
import numpy as np
from PIL import Image
import cv2  # For template matching
import os
import keyboard

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# === Load template image ===
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "screenshot.png")
template_img = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
if template_img is None:
    raise FileNotFoundError(f"Template image not found at {TEMPLATE_PATH}")
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
template_w, template_h = template_gray.shape[::-1]

# Add this near the top of the file, after target definitions
click_counts = {
    "Claim": 0,
    "Retry": 0,
    "Sigil": 0,
}


def save_screenshot(image, tag=""):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("screenshots", f"{tag}_{timestamp}.png")
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")


def detect_sigil_from_templates(image, template_dir="sigil_templates", threshold=0.650):
    """Detects sigil in image using multiple rotated templates. Returns (x, y) or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_score = 0

    for template_path in glob.glob(os.path.join(template_dir, "*.png")):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Failed to load template: {template_path}")
            continue

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_match = (max_loc, template.shape[::-1])  # ((x, y), (w, h))

    if best_match and best_score >= 0.55:
        # Define point and radius
        (x, y), (w, h) = best_match
        center_x = x + w // 2
        center_y = y + h // 2

        c_center = (center_x, center_y)  # (x, y) coordinates of the point
        c_radius = 40  # Circle radius in pixels
        c_color = (0, 255, 0)  # Green in BGR
        c_thickness = 2  # Line thickness (-1 for filled circle)

        cv2.circle(image, c_center, c_radius, c_color, c_thickness)

        save_screenshot(image, 'S_TEST')

    if best_match and best_score >= threshold:
        (x, y), (w, h) = best_match
        center_x = x + w // 2
        center_y = y + h // 2
        print(f"Best match: {center_x},{center_y} with score {best_score:.3f}")
        return center_x, center_y

    print(f"No template match found above threshold. Best score: {best_score:.3f}")
    return None


def get_phone_window(title_contains="The Tower"):
    """Find the Phone Link window."""
    windows = gw.getWindowsWithTitle(title_contains)
    return windows[0] if windows else None


def capture_window(win):
    """Capture a screenshot of the given window using mss (multi-monitor compatible)."""
    bbox = {
        "top": win.top,
        "left": win.left,
        "width": win.width,
        "height": win.height
    }

    with mss.mss() as sct:
        sct_img = sct.grab(bbox)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return np.array(img)


def detect_text(img):
    """Run EasyOCR and return list of (text, confidence, bbox)."""
    img_array = np.array(img)
    results = reader.readtext(img_array)
    return [(text, conf, bbox) for (bbox, text, conf) in results]


def click_relative(win, x_rel, y_rel, return_mouse=True):
    """Click inside the window at a relative position. Optionally return mouse to original position."""
    target_x = win.left + int(win.width * x_rel)
    target_y = win.top + int(win.height * y_rel)

    if return_mouse:
        original_pos = pyautogui.position()

    pyautogui.click(target_x, target_y)

    if return_mouse:
        pyautogui.moveTo(original_pos)




# === Main Loop ===
if __name__ == "__main__":
    window = get_phone_window()

    if not window:
        print("Could not find Phone Link window.")
        exit()

    # Define the text triggers and click locations
    targets = {
        "Claim": (0.1, 0.6),  # Adjust for actual button location
        "Retry": (0.4, 0.7),  # Adjust for actual Retry location
    }

    print("Looking for buttons or sigil box...")

    frame = 0

    while True:
        img = capture_window(window)
        results = detect_text(img)

        matched = False

        if frame % 3 == 0:
            for text, conf, _ in results:
                for target_text, (x_rel, y_rel) in targets.items():
                    if target_text.lower() in text.lower() and conf >= 0.5:
                        print(f"> Clicking on '{target_text}' match: '{text}'")
                        save_screenshot(img, target_text)
                        click_relative(window, x_rel, y_rel)
                        click_counts[target_text] += 1
                        matched = True
                        time.sleep(2)
                        break
                if matched:
                    break

        # Second: Try sigil detection if no text matched
        sigil_pos = detect_sigil_from_templates(img)
        if sigil_pos:
            abs_x = window.left + sigil_pos[0]
            abs_y = window.top + sigil_pos[1]
            print(f"> Clicking on detected sigil box at ({abs_x}, {abs_y})")

            center = (sigil_pos[0], sigil_pos[1])  # (x, y) coordinates of the point
            radius = 40  # Circle radius in pixels
            color = (255, 255, 0)  # Green in BGR
            thickness = 2  # Line thickness (-1 for filled circle)

            cv2.circle(img, center, radius, color, thickness)
            save_screenshot(img, 'SIGIL_CONFIRM')

            pyautogui.click(abs_x, abs_y)
            click_counts["Sigil"] += 1
            time.sleep(2)
            matched = True

        # Report click counts on failed attempt
        if not matched:
            print("No match found. Retrying...")
            print("Click counters:")
            for k, v in click_counts.items():
                print(f"  {k}: {v}")

        frame = frame + 1

        if frame > 12:
            frame = 0