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

    height = image.shape[0]
    low = int(height * 0.4)
    high = int(height * 0.2)


    image = image[low:, high:]

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

        cv2.putText(
            image,  # the image to draw on
            f"{best_score:.3f}",  # the text string
            (center_x - 20, center_y - 60),  # bottom-left corner of the text
            cv2.FONT_HERSHEY_SIMPLEX,  # font type
            0.7,  # font scale (size)
            (0, 0, 255),  # color in BGR (white here)
            2,  # thickness
            cv2.LINE_AA  # line type (anti-aliased for better quality)
        )

        cv2.circle(image, c_center, c_radius, c_color, c_thickness)

        save_screenshot(image, 'S_TEST2')

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


    original_pos = pyautogui.position()

    pyautogui.click(target_x, target_y)

    if return_mouse:
        pyautogui.moveTo(original_pos)

def detect_upgrade_templates(image, template_dir="upgrade_template", debug=True, threshold=0.6):
    if image is None:
        raise ValueError("Input image is None")

    height = image.shape[0]
    lower_third_start = int(height * 2 / 3)
    lower_third = image[lower_third_start:, :]

    debug_image = image.copy()
    boxes = []
    confidences = []

    for filename in os.listdir(template_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        template_path = os.path.join(template_dir, filename)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            continue

        res = cv2.matchTemplate(lower_third, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            full_y = pt[1] + lower_third_start
            box = [pt[0], full_y, 200, 80]  # x, y, w, h
            conf = float(res[pt[1], pt[0]])

            boxes.append(box)
            confidences.append(conf)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=0.3)

    results = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            conf = confidences[i]

            # Draw bounding box for the upgrade panel
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(debug_image, f"{conf:.2f}", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            roi = image[y:y + h, x:x + w]
            ocr_result = reader.readtext(roi)

            upgrade_name = ""
            value = ""
            cost = ""

            for bbox, text, conf in ocr_result:
                (tl, tr, br, bl) = bbox
                cx = int((tl[0] + br[0]) / 2)
                cy = int((tl[1] + br[1]) / 2)

                label = ""
                if cx < w / 2:
                    upgrade_name += text + " "
                    label = "name"
                else:
                    # Decide whether it's value or cost based on vertical position
                    if cy < h / 2:
                        value = text
                        label = "value"
                    else:
                        cost = text
                        label = "cost"

                # Draw debug boxes and label
                cv2.rectangle(debug_image, (x + int(tl[0]), y + int(tl[1])), (x + int(br[0]), y + int(br[1])), (255, 0, 0), 1)
                cv2.putText(debug_image, label, (x + int(tl[0]), y + int(tl[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            results.append((upgrade_name.strip(), value.strip(), cost.strip()))

    if debug:
        os.makedirs("upgrade_test", exist_ok=True)
        filename = "debug_labeled_output.jpg"
        debug_path = os.path.join("upgrade_test", filename)
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug image saved to {debug_path}")

    return results


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

        for t in detect_upgrade_templates(img):
           print(f"{t[0], t[1], t[2]}")

        if frame % 3 == 0:
            for text, conf, _ in results:
                for target_text, (x_rel, y_rel) in targets.items():
                    if target_text.lower() in text.lower() and conf >= 0.5:
                        print(f"> Clicking on '{target_text}' match: '{text}'")
                        save_screenshot(img, target_text)
                        click_relative(window, x_rel, y_rel)
                        click_counts[target_text] += 1
                        matched = True
                        time.sleep(0.2)
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

            #save mouse position, move and click, then return mouse position
            original_pos = pyautogui.position()
            pyautogui.click(abs_x, abs_y)
            pyautogui.moveTo(original_pos)

            click_counts["Sigil"] += 1
            time.sleep(2)
            matched = True

        # Report click counts on failed attempt
        #if not matched:
            #print("No match found. Retrying...")
            #print("Click counters:")
            #for k, v in click_counts.items():
                #print(f"  {k}: {v}")

        frame = frame + 1

        if frame > 12:
            frame = 0