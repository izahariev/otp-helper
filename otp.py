import argparse
import sqlite3

import cv2
import numpy as np
from PIL import ImageGrab
from skimage.metrics import structural_similarity as ssim


def sanitize_image(img, img_index):
    screenshot_arr = np.array(img)

    (h, w) = screenshot_arr.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the new bounding dimensions to ensure the entire image is retained after rotation
    new_w = int(w * np.abs(np.cos(np.radians(30))) + h * np.abs(np.sin(np.radians(30))))
    new_h = int(h * np.abs(np.cos(np.radians(30))) + w * np.abs(np.sin(np.radians(30))))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix = cv2.getRotationMatrix2D(center, -30, 1.0)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Perform the rotation without cutting parts of the image
    rotated_image = cv2.warpAffine(screenshot_arr, rotation_matrix, (new_w, new_h))

    match img_index:
        case 1:
            return rotated_image[68:-64, 14:-15, :]
        case 2:
            return rotated_image[67:-65, 13:-15, :]
        case 3:
            return rotated_image[68:-60, 12:-14, :]
        case 4:
            return rotated_image[60:-65, 5:-13, :]
        case 5:
            return rotated_image[66:-64, 14:-13, :]


def scan():
    # List of coordinates for each name position
    names_coordinates = [
        [1783, 180, 1910, 275],
        [1689, 350, 1815, 445],
        [1783, 520, 1910, 610],
        [1700, 690, 1815, 784],
        [1783, 860, 1910, 952]
    ]

    # Connect to the database
    conn = sqlite3.connect('db/otp.db')
    cursor = conn.cursor()
    cursor.execute('SELECT player_name, heroes_info, image, width, height FROM otp')
    rows = cursor.fetchall()
    conn.close()

    i = 1
    for coordinates in names_coordinates:
        with ImageGrab.grab(bbox=(tuple(coordinates))) as img:
            # Convert image to white text on black background
            sanitized_img = sanitize_image(img, i)
            sanitized_img_gray = cv2.cvtColor(sanitized_img, cv2.COLOR_BGR2GRAY)

            # Iterate over database entries and compare images
            for row in rows:
                player_name, heroes_info, db_image_data, width, height = row
                db_image_array = np.frombuffer(db_image_data, dtype=np.uint8)
                db_image = db_image_array.reshape((height, width))

                # Resize sanitized image if necessary
                if db_image.shape != sanitized_img_gray.shape:
                    sanitized_img_resized = cv2.resize(sanitized_img_gray, (db_image.shape[1], db_image.shape[0]))
                else:
                    sanitized_img_resized = sanitized_img_gray

                # Calculate SSIM between the two images
                similarity_index, _ = ssim(db_image, sanitized_img_resized, full=True)

                # If the SSIM is above a threshold, consider it a match
                if similarity_index > 0.9:
                    print("=======================================================================")
                    print(player_name)
                    print(heroes_info)
                    print("=======================================================================")

            # Save image to file system
            cv2.imwrite('images/' + str(i) + '.png', sanitized_img)
        i += 1


def add(player_index, player_name, heroes_info_list):
    img = cv2.imread(str(player_index) + '.png', cv2.IMREAD_GRAYSCALE)
    img_data = img.tobytes()

    # Connect to the database (or create it if it doesn't exist)
    conn = sqlite3.connect('db/otp.db')
    cursor = conn.cursor()

    # Check if the 'otp' table exists, and create it if it doesn't
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS otp (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT,
            heroes_info TEXT,
            image BLOB,
            width INTEGER,
            height INTEGER
        )
    ''')

    # Insert the new entry into the 'otp' table
    heroes_info_str = '\n'.join([f"{value} - {key}" for key, value in heroes_info_list.items()])
    cursor.execute(
        'INSERT INTO otp (player_name, heroes_info, image, width, height) VALUES (?, ?, ?, ?, ?)',
        (player_name, heroes_info_str, sqlite3.Binary(img_data), img.shape[1], img.shape[0])
    )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Image processing program with scan and add options.")
    subparsers = parser.add_subparsers(dest="command")

    # Scan command
    subparsers.add_parser("scan", help="Scan images and process them.")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add entry to the database.")
    add_parser.add_argument("player_index", type=int, help="Player position in the enemy team")
    add_parser.add_argument("player_name", type=str, help="Player name")
    add_parser.add_argument("heroes_info", type=str,
                            help="Comma-separated list of heroes info. (ex.: P0-Zul Jin,P1- Cho Gal, P0-Kel'Tuzad)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "scan":
        scan()
    elif args.command == "add":
        # Parse heroes info into a dictionary
        heroes_info_list = args.heroes_info.split(",")
        heroes_info_dict = {}
        for element in heroes_info_list:
            value, key = element.split("-", 1)
            heroes_info_dict[key.strip()] = value.strip()

        add(args.player_index, args.player_name, heroes_info_dict)


if __name__ == "__main__":
    main()
