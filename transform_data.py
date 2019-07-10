import cv2
import os
import numpy as np
from Rectangle import Rectangle

DATA_PATH_SRC = "/home/stefan/DataSets/chars_and_numbers"
DATA_PATH_DST = "/home/stefan/DataSets/chars_and_numbers_1"


# due to some thin fonts, we apply this in order to have all parts of character connected, so
# it can be detected as single contour, but with minimal changes from original character
def dilate_and_erode(img):
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    return img


# called for characters consisted of two parts (i.e. "i", "j",";",":")
def merge_bounding_rects(contour_list):
    r1 = Rectangle(cv2.boundingRect(contour_list[0]))
    r2 = Rectangle(cv2.boundingRect(contour_list[1]))
    left = r1.get_left() if r1.get_left() < r2.get_left() else r2.get_left()
    top = r1.get_top() if r1.get_top() < r2.get_top() else r2.get_top()
    right = r1.get_right() if r1.get_right() > r2.get_right() else r2.get_right()
    bottom = r1.get_bottom() if r1.get_bottom() > r2.get_bottom() else r2.get_bottom()
    return [left, top, right, bottom]


# this function removes white padding from all sides of character
def remove_padding(new_img):
    img = new_img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    binary_img = dilate_and_erode(binary_img)
    contour_list, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left, top, right, bottom = 0, 0, 0, 0
    if len(contour_list) == 2:
        [left, top, right, bottom] = merge_bounding_rects(contour_list)
    else:
        [left, top, width, height] = cv2.boundingRect(contour_list[0])
        right, bottom = left + width, top + height
    return new_img[top:bottom, left:right]


os.mkdir(DATA_PATH_DST)
for folder in (os.listdir(DATA_PATH_SRC)):
    path = os.path.join(DATA_PATH_SRC, folder)
    if not os.path.isdir(path):
        continue

    os.mkdir(os.path.join(DATA_PATH_DST, folder))
    for file in (os.listdir(path)):
        if file.endswith('.png'):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            img = remove_padding(img)
            img = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(DATA_PATH_DST, folder, file), img)
