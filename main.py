import cv2
import numpy as np
import pytesseract

from PIL import ImageGrab


def extract_player_name(img, max_hue):
    """
    Extracts the players name from the given image. The method does the following to achieve that:
        - rotate the image 30 degrees to the right. This is done because the text recognition is greatly improved
          if the text is horizontal
        - zoom the image
        - create a matrix (mask) with the coordinates of the pixels which color is within the range
          HSV(277, 20%, 39%) - HSV({max_hue}, 23%, 67%)
        - create a white image and apply the mask on top of it making the pixels with coordinates in the mask black. As
          a result of this the image is converted to black text on white background
        - use PyTesseract to read the text from the transformed image
    :param img: ImageGrab image
    :param max_hue: the max Hue of the colors which should be converted to black. This is needed because for the first
        position from the top the optimal max Hue is 281, but for the other positions is 282.
    :return: The extracted name
    """
    # Convert the image from RGB (PIL) to BGR (OpenCV)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Rotate the image 30 degrees to the right
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the new bounding dimensions to ensure the entire image is retained after rotation
    new_w = int(w * np.abs(np.cos(np.radians(30))) + h * np.abs(np.sin(np.radians(30))))
    new_h = int(h * np.abs(np.cos(np.radians(30))) + w * np.abs(np.sin(np.radians(30))))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix = cv2.getRotationMatrix2D(center, -30, 1.0)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Perform the rotation without cutting parts of the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    # Zoom factor
    zoom_factor = 5  # Increase or decrease this value to zoom in or out
    zoomed_image = cv2.resize(rotated_image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    # Convert the rotated image to HSV color space
    hsv = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2HSV)

    # Define the HSV color range for the text
    lower_hsv = np.array([277 / 2, 20 / 100 * 255, 39 / 100 * 255], dtype=np.uint8)  # Convert to OpenCV scale
    upper_hsv = np.array([max_hue / 2, 23 / 100 * 255, 67 / 100 * 255], dtype=np.uint8)  # Convert to OpenCV scale
    # 281, 282

    # Create a mask for the text color range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Create a white image
    final_image = np.full_like(zoomed_image, 255)  # Start with a white image

    # Set the pixels within the mask to black
    final_image[mask > 0] = [0, 0, 0]  # Set the detected text areas to black

    # Extract text using pytesseract
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(final_image, config=custom_config)

    names = []
    for name in text.split("\n"):
        # For some names pytesseract mistakenly guesses that it contains special characters or spaces.
        # These characters are not allowed in names so it is a mistake. That's why we are removing them
        sanitized_name = name.replace("|", "").replace(" ", "").replace(".", "")
        names.append(sanitized_name)

    # Sometimes there are wrongly interpreted characters after the name. They are always after the name so we are
    # just going to ignore them
    return names[0]


def main():
    # List of coordinates and max Hue for each name position
    names_data = [
        {
            "coordinates": [1783, 180, 1910, 275],
            "max_hue": 281
        },
        {
            "coordinates": [1689, 350, 1815, 445],
            "max_hue": 282
        },
        {
            "coordinates": [1783, 520, 1910, 610],
            "max_hue": 282
        },
        {
            "coordinates": [1700, 690, 1815, 784],
            "max_hue": 282
        },
        {
            "coordinates": [1783, 860, 1910, 952],
            "max_hue": 282
        }
    ]

    for data in names_data:
        # Capture the screenshot
        with ImageGrab.grab(bbox=(tuple(data["coordinates"]))) as screenshot:
            print(extract_player_name(screenshot, data["max_hue"]))


if __name__ == "__main__":
    main()
