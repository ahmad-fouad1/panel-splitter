import cv2
import os

from main2Base64 import split_panels


# -----------------------------
# Load test image
# -----------------------------

image_path = r"C:\Users\ahmed\Downloads\panel-splitter-api\panel-splitter-api\resources\ChatGPT Image Aug 31, 2026, 02_22_49 PM.png"

image = cv2.imread(image_path)

if image is None:
    raise Exception(f"Could not load image: {image_path}")


print("Image loaded successfully")

height, width = image.shape[:2]

print(f"Image size: {width} x {height}")


# -----------------------------
# Split image
# -----------------------------

try:

    panels = split_panels(image)

except Exception as e:

    print("\nERROR:")
    print(e)

    raise


# -----------------------------
# Create test output folder
# -----------------------------

os.makedirs("test_output", exist_ok=True)


# -----------------------------
# Save panels
# -----------------------------

for panel_number, panel in panels:

    output_path = f"test_output/panel_{panel_number}.jpg"

    cv2.imwrite(
        output_path,
        panel
    )

    h, w = panel.shape[:2]

    print(
        f"Panel {panel_number}: "
        f"{w} x {h} -> {output_path}"
    )


print("\nSUCCESS!")
print("6 panels were extracted.")