import cv2
import json
import numpy as np
import pytesseract
from mss import mss
from PIL import Image

grid_box = {"top": 124, "left": 521, "width": 220, "height": 518}  # grid
frame_box = {"top": 748, "left": 640, "width": 105, "height": 25}  # frame

held_box = {"top": 220, "left": 413, "width": 90, "height": 60}  # held piece
next_box = {"top": 220, "left": 772, "width": 90, "height": 330}  # next pieces

lower_cyan = np.array([57, 113, 88])
upper_cyan = np.array([158, 250, 203])

lower_blue = np.array([62, 53, 108])
upper_blue = np.array([154, 144, 224])

lower_orange = np.array([152, 94, 53])
upper_orange = np.array([245, 173, 113])

lower_yellow = np.array([176, 156, 63])
upper_yellow = np.array([236, 217, 146])

lower_green = np.array([137, 177, 66])
upper_green = np.array([201, 238, 135])

lower_purple = np.array([100, 52, 100])
upper_purple = np.array([217, 148, 215])

lower_red = np.array([149, 58, 57])
upper_red = np.array([240, 122, 120])

sct = mss()


def get_frame():
    img = np.array(sct.grab(frame_box))

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(gray_img)

    # Extract the frame number from the OCR text
    text = "".join(filter(str.isdigit, text))

    frame = int(text) if text else 0
    return frame


def get_grid():
    img = np.array(sct.grab(grid_box))

    x_off = 22
    y_off = 23
    grid = np.empty((23, 10), dtype=str)

    for row in range(23):
        for col in range(10):
            left = col * x_off + 9
            top = row * y_off + 10
            if row > 9:
                top -= 1 * (row - 10)

            color = img[top, left]
            color = np.array([color[2], color[1], color[0]])

            if np.all(color >= lower_cyan) and np.all(color <= upper_cyan):
                grid[row, col] = "I"
            elif np.all(color >= lower_blue) and np.all(color <= upper_blue):
                grid[row, col] = "J"
            elif np.all(color >= lower_yellow) and np.all(color <= upper_yellow):
                grid[row, col] = "O"
            elif np.all(color >= lower_green) and np.all(color <= upper_green):
                grid[row, col] = "S"
            elif np.all(color >= lower_purple) and np.all(color <= upper_purple):
                grid[row, col] = "T"
            elif np.all(color >= lower_orange) and np.all(color <= upper_orange):
                grid[row, col] = "L"
            elif np.all(color >= lower_red) and np.all(color <= upper_red):
                grid[row, col] = "Z"
            else:
                grid[row, col] = "."

            # cv2.circle(img, (left, top), radius=0, color=(0, 0, 255), thickness=1)

    return grid


def get_next():
    detected = []
    img = np.array(sct.grab(next_box))

    for row_y in [39, 106, 174, 241, 308]:
        I = img[row_y, 31]
        I = np.array([I[2], I[1], I[0]])

        O = img[row_y, 31]
        O = np.array([O[2], O[1], O[0]])

        Z = img[row_y, 42]
        Z = np.array([Z[2], Z[1], Z[0]])

        TLJS = img[row_y, 21]
        TLJS = np.array([TLJS[2], TLJS[1], TLJS[0]])

        if np.all(I >= lower_cyan) and np.all(I <= upper_cyan):
            detect = "I"
        elif np.all(TLJS >= lower_blue) and np.all(TLJS <= upper_blue):
            detect = "J"
        elif np.all(O >= lower_yellow) and np.all(O <= upper_yellow):
            detect = "O"
        elif np.all(TLJS >= lower_green) and np.all(TLJS <= upper_green):
            detect = "S"
        elif np.all(TLJS >= lower_purple) and np.all(TLJS <= upper_purple):
            detect = "T"
        elif np.all(TLJS >= lower_orange) and np.all(TLJS <= upper_orange):
            detect = "L"
        elif np.all(Z >= lower_red) and np.all(Z <= upper_red):
            detect = "Z"
        else:
            detect = "."

        detected.append(detect)

    return detected


def is_active_pixel(rgb):
    return rgb[0] > 0 or rgb[1] > 0 or rgb[2] > 0


def is_gray_pixel(rgb):
    return rgb[0] <= 55 and rgb[1] <= 55 and rgb[2] <= 55 and rgb[0] > 0 and rgb[1] > 0 and rgb[2] > 0


def get_held():
    img = np.array(sct.grab(held_box))

    TL = img[9, 14]
    TL = np.array([TL[2], TL[1], TL[0]])
    TM = img[9, 36]
    TM = np.array([TM[2], TM[1], TM[0]])
    TR = img[9, 58]
    TR = np.array([TR[2], TR[1], TR[0]])
    BL = img[31, 14]
    BL = np.array([BL[2], BL[1], BL[0]])
    BM = img[31, 36]
    BM = np.array([BM[2], BM[1], BM[0]])
    BR = img[31, 58]
    BR = np.array([BR[2], BR[1], BR[0]])

    active_pixels = [1 if is_active_pixel(val) else 0 for val in [TL, TM, TR, BL, BM, BR]]
    gray_pixels = [1 if is_gray_pixel(val) else 0 for val in [TL, TM, TR, BL, BM, BR]]
    held = "None"

    if active_pixels == [0, 0, 0, 0, 0, 0]:
        return "None", True
    elif active_pixels == [0, 0, 0, 1, 1, 1]:
        held = "I"
    elif active_pixels == [1, 0, 0, 1, 1, 1]:
        held = "J"
    elif active_pixels == [0, 0, 1, 1, 1, 1]:
        held = "L"
    elif active_pixels == [0, 1, 1, 0, 1, 1]:
        held = "O"
    elif active_pixels == [0, 1, 1, 1, 1, 0]:
        held = "S"
    elif active_pixels == [0, 1, 0, 1, 1, 1]:
        held = "T"
    elif active_pixels == [1, 1, 0, 0, 1, 1]:
        held = "Z"

    # cv2.circle(img, (14, 9), radius=0, color=(0, 0, 255), thickness=1)
    # cv2.circle(img, (36, 9), radius=0, color=(0, 0, 255), thickness=1)
    # cv2.circle(img, (58, 9), radius=0, color=(0, 0, 255), thickness=1)
    # cv2.circle(img, (14, 31), radius=0, color=(0, 0, 255), thickness=1)
    # cv2.circle(img, (36, 31), radius=0, color=(0, 0, 255), thickness=1)
    # cv2.circle(img, (58, 31), radius=0, color=(0, 0, 255), thickness=1)
    # cv2.imshow("Held", img)

    return held, (active_pixels != gray_pixels)


def create_game_state(frame_number, game_board, current_piece, held_piece, holdable, get_next_pieces):
    game_state = {
        "frame": frame_number,
        "game_board": game_board,
        "current_piece": current_piece,
        "held_piece": held_piece,
        "can_hold": holdable,
        "next_pieces": get_next_pieces,
    }
    return game_state


def get_current_piece(grid, prev_state):
    buffer_zone = grid[:3]
    values = np.unique(buffer_zone)

    if len(values) == 1 and values[0] == ".":
        return prev_state["current_piece"] if prev_state else "None"
    else:
        return values[1].item()


def screen_record():
    states = []

    while True:
        frame = get_frame()
        if frame == 0:
            continue
        next = get_next()
        grid = get_grid()
        held, avail = get_held()
        curr = get_current_piece(grid, states[-1] if states else None)
        state = create_game_state(frame, grid.tolist(), curr, held, avail, next)
        states.append(state)

        print(f"{grid} \n Frame: {frame} \n Current: {curr} \n Held: {held} Holdable: {avail} \n Next: {next} \n")

        if frame >= 7199:
            break
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

    # Save the training data to a JSON file
    with open("training_data.json", "w") as file:
        json.dump(states, file)


def main():
    screen_record()


if __name__ == "__main__":
    main()
