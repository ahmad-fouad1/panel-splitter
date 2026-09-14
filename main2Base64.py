from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64


app = FastAPI()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "panel-splitter-api"
    }


# ============================================================
# GROUP CONTINUOUS LINES
# ============================================================

def group_continuous_lines(indices):
    """
    Group neighboring pixel indexes into individual lines.
    """

    if len(indices) == 0:
        return []

    groups = []

    start = indices[0]
    previous = indices[0]

    for index in indices[1:]:

        if index <= previous + 1:
            previous = index

        else:
            groups.append(
                (start, previous)
            )

            start = index
            previous = index

    groups.append(
        (start, previous)
    )

    return groups


# ============================================================
# DETECT DIVIDER LINES
# ============================================================

def detect_divider_lines(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # --------------------------------------------------------
    # Detect bright / white pixels
    # --------------------------------------------------------

    white_mask = cv2.inRange(
        gray,
        235,
        255
    )

    # --------------------------------------------------------
    # Calculate white pixel ratio
    # --------------------------------------------------------

    vertical_ratio = np.mean(
        white_mask == 255,
        axis=0
    )

    horizontal_ratio = np.mean(
        white_mask == 255,
        axis=1
    )

    # --------------------------------------------------------
    # Ignore image borders
    # --------------------------------------------------------

    edge_margin_x = int(width * 0.05)
    edge_margin_y = int(height * 0.05)

    if edge_margin_x > 0:

        vertical_ratio[:edge_margin_x] = 0
        vertical_ratio[-edge_margin_x:] = 0

    if edge_margin_y > 0:

        horizontal_ratio[:edge_margin_y] = 0
        horizontal_ratio[-edge_margin_y:] = 0

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    vertical_candidates = np.where(
        vertical_ratio >= 0.60
    )[0]

    horizontal_candidates = np.where(
        horizontal_ratio >= 0.60
    )[0]

    # --------------------------------------------------------
    # Group continuous pixels
    # --------------------------------------------------------

    vertical_groups = group_continuous_lines(
        vertical_candidates
    )

    horizontal_groups = group_continuous_lines(
        horizontal_candidates
    )

    # --------------------------------------------------------
    # Convert groups to centers
    # --------------------------------------------------------

    vertical_lines = [
        (start + end) // 2
        for start, end in vertical_groups
    ]

    horizontal_lines = [
        (start + end) // 2
        for start, end in horizontal_groups
    ]

    return (
        vertical_lines,
        horizontal_lines
    )


# ============================================================
# FIND BEST VERTICAL DIVIDER
# ============================================================

def find_best_vertical_divider(
    lines,
    width
):

    if not lines:

        raise ValueError(
            "No vertical divider detected."
        )

    # Expected divider is approximately
    # in the center of the image.

    center = width / 2

    best = min(
        lines,
        key=lambda x: abs(x - center)
    )

    return best


# ============================================================
# FIND BEST HORIZONTAL DIVIDERS
# ============================================================

def find_best_horizontal_dividers(
    lines,
    height
):

    if not lines:

        raise ValueError(
            "No horizontal dividers detected."
        )

    # Expected horizontal dividers are approximately:
    #
    # 1/3 of image height
    # 2/3 of image height

    expected = [
        height / 3,
        height * 2 / 3
    ]

    selected = []

    remaining = list(lines)

    for target in expected:

        if not remaining:
            break

        best = min(
            remaining,
            key=lambda x: abs(x - target)
        )

        selected.append(best)

        remaining.remove(best)

    if len(selected) != 2:

        raise ValueError(
            "Could not find 2 horizontal dividers. "
            f"Detected: {lines}"
        )

    return sorted(selected)


# ============================================================
# SPLIT PANELS
# ============================================================

def split_panels(image):

    height, width = image.shape[:2]

    print(
        f"Image dimensions: {width} x {height}"
    )

    # --------------------------------------------------------
    # Detect lines
    # --------------------------------------------------------

    (
        vertical_lines,
        horizontal_lines
    ) = detect_divider_lines(image)

    print(
        "Detected vertical lines:",
        vertical_lines
    )

    print(
        "Detected horizontal lines:",
        horizontal_lines
    )

    # --------------------------------------------------------
    # Find real vertical divider
    # --------------------------------------------------------

    vertical = find_best_vertical_divider(
        vertical_lines,
        width
    )

    # --------------------------------------------------------
    # Find horizontal dividers
    # --------------------------------------------------------

    horizontal = find_best_horizontal_dividers(
        horizontal_lines,
        height
    )

    horizontal_1 = horizontal[0]
    horizontal_2 = horizontal[1]

    print(
        "Selected vertical divider:",
        vertical
    )

    print(
        "Selected horizontal dividers:",
        horizontal_1,
        horizontal_2
    )

    # --------------------------------------------------------
    # Calculate boundaries
    # --------------------------------------------------------

    x_boundaries = [
        0,
        vertical,
        width
    ]

    y_boundaries = [
        0,
        horizontal_1,
        horizontal_2,
        height
    ]

    panels = []

    panel_number = 1

    # --------------------------------------------------------
    # Extract 6 panels
    # --------------------------------------------------------

    for row in range(3):

        y1 = y_boundaries[row]
        y2 = y_boundaries[row + 1]

        for column in range(2):

            x1 = x_boundaries[column]
            x2 = x_boundaries[column + 1]

            # ------------------------------------------------
            # Remove divider line
            # ------------------------------------------------

            margin = 2

            if column == 0:

                x2 = max(
                    x1 + 1,
                    x2 - margin
                )

            else:

                x1 = min(
                    x2 - 1,
                    x1 + margin
                )

            if row == 0:

                y2 = max(
                    y1 + 1,
                    y2 - margin
                )

            else:

                y1 = min(
                    y2 - 1,
                    y1 + margin
                )

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            panel = image[
                y1:y2,
                x1:x2
            ]

            if panel.size == 0:

                raise ValueError(
                    f"Panel {panel_number} is empty."
                )

            panels.append(
                (
                    panel_number,
                    panel
                )
            )

            panel_number += 1

    return panels


# ============================================================
# CONVERT IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    success, encoded = cv2.imencode(
        ".png",
        image
    )

    if not success:

        raise ValueError(
            "Could not encode image."
        )

    encoded_bytes = encoded.tobytes()

    base64_image = base64.b64encode(
        encoded_bytes
    ).decode("utf-8")

    return base64_image


# ============================================================
# SPLIT API
# ============================================================

@app.post("/split")
async def split_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be detected."
        )

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a valid image file "
                "(JPG, JPEG, PNG, or WEBP)."
            )
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image = cv2.imdecode(
        np.frombuffer(
            contents,
            dtype=np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Split image
    # --------------------------------------------------------

    try:

        panels = split_panels(image)

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # --------------------------------------------------------
    # Make sure we have exactly 6 panels
    # --------------------------------------------------------

    if len(panels) != 6:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Expected 6 panels, "
                f"but got {len(panels)}."
            )
        )

    # --------------------------------------------------------
    # Prepare response
    # --------------------------------------------------------

    result = []

    for panel_number, panel in panels:

        try:

            image_base64 = image_to_base64(
                panel
            )

        except ValueError as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not encode "
                    f"panel {panel_number}: {str(e)}"
                )
            )

        result.append(
            {
                "panel": panel_number,
                "filename": (
                    f"panel_{panel_number}.png"
                ),
                "mime_type": "image/png",
                "image": image_base64
            }
        )

    # --------------------------------------------------------
    # Return images
    # --------------------------------------------------------

    return JSONResponse(
        content={
            "status": "success",
            "count": len(result),
            "panels": result
        }
    )