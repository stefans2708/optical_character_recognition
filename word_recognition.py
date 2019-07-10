import numpy as np
import cv2
import constants
from Rectangle import Rectangle
from keras.models import model_from_json


#############################
# FUNCTIONS
#############################

def load_model_data():
    json_file = open(constants.MODEL_ARCH_FILE_NAME, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(constants.MODEL_WEIGHTS_FILE_NAME)
    return loaded_model


def merge_multipart_chars(boundaries):
    result_boundaries = []
    i = 0
    while i < (len(boundaries)):
        if i == len(boundaries) - 1:
            result_boundaries.append(boundaries[i])
            break
        if boundaries[i].contains_point_on_x_axcis(boundaries[i + 1].left):
            boundaries[i].set_top(boundaries[i].top if boundaries[i].top < boundaries[i + 1].top else boundaries[i + 1].top)
            boundaries[i].set_bottom(boundaries[i].bottom if boundaries[i].bottom > boundaries[i + 1].bottom else boundaries[i + 1].bottom)
            boundaries[i].set_right(boundaries[i].right if boundaries[i].right > boundaries[i + 1].right else boundaries[i + 1].right)
            result_boundaries.append(boundaries[i])
            i += 2
        else:
            result_boundaries.append(boundaries[i])
            i += 1
    return result_boundaries


def read_image_as_binary(img_path):
    original_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)  # must be inverted
    return binary_img


def recognize(img_path, model):
    extracted_char_imgs = []
    bounding_rectangles = []

    binary_img = read_image_as_binary(img_path)
    contour_list, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contour_list:
        bounding_rectangles.append(Rectangle(cv2.boundingRect(contour)))

    bounding_rectangles.sort(key=lambda img_char: img_char.get_left())

    bounding_rectangles = merge_multipart_chars(bounding_rectangles)

    for rect in bounding_rectangles:
        char_img = binary_img[rect.get_top():rect.get_bottom(), rect.get_left():rect.get_right()]
        char_img = cv2.resize(char_img, (constants.IMG_SIZE, constants.IMG_SIZE), interpolation=cv2.INTER_AREA)
        char_img = char_img.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE)
        char_img = ~char_img  # re-invert it back to normal
        extracted_char_imgs.append(char_img)

    predicted = ""
    for i in range(len(extracted_char_imgs)):
        y_out = model.predict([extracted_char_imgs[i]])
        y_out = np.argmax(y_out, axis=1)
        predicted += constants.CHARS[y_out[0]]

    return predicted

#############################
# MAIN
#############################

# predicted = ""
# model = load_model_data()
# original_img = cv2.imread("input_data/najjaci.png")
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)  # must be inverted
# contour_list, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# extracted_char_imgs = []
# bounding_rectangles = []
#
# for contour in contour_list:
#     bounding_rectangles.append(Rectangle(cv2.boundingRect(contour)))
#
# bounding_rectangles.sort(key=lambda img_char: img_char.get_left())
#
# bounding_rectangles = merge_multipart_chars(bounding_rectangles)
#
# for rect in bounding_rectangles:
#     char_img = binary_img[rect.get_top():rect.get_bottom(), rect.get_left():rect.get_right()]
#     char_img = cv2.resize(char_img, (constants.IMG_SIZE, constants.IMG_SIZE), interpolation=cv2.INTER_AREA)
#     char_img = char_img.reshape(-1, constants.IMG_SIZE, constants.IMG_SIZE)
#     char_img = ~char_img  # re-invert it back to normal
#     extracted_char_imgs.append(char_img)
#
# for i in range(len(extracted_char_imgs)):
#     y_out = model.predict([extracted_char_imgs[i]])
#     y_out = np.argmax(y_out, axis=1)
#     predicted += constants.CHARS[y_out[0]]
#
# print(predicted)
