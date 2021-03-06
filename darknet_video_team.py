from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

color_dict_default = {
    'person': [0, 128, 0], 'bicycle': [238, 123, 158], 'car': [24, 245, 217], 'motorbike': [224, 119, 227],
    'aeroplane': [154, 52, 104], 'bus': [179, 50, 247], 'train': [180, 164, 5], 'truck': [82, 42, 106],
    'boat': [201, 25, 52], 'traffic light': [62, 17, 209], 'fire hydrant': [60, 68, 169], 'stop sign': [199, 113, 167],
    'parking meter': [19, 71, 68], 'bench': [161, 83, 182], 'bird': [75, 6, 145], 'cat': [100, 64, 151],
    'dog': [156, 116, 171], 'horse': [88, 9, 123], 'sheep': [181, 86, 222], 'cow': [116, 238, 87],
    'elephant': [74, 90, 143], 'bear': [249, 157, 47], 'zebra': [26, 101, 131], 'giraffe': [195, 130, 181],
    'backpack': [242, 52, 233], 'umbrella': [131, 11, 189], 'handbag': [221, 229, 176], 'tie': [193, 56, 44],
    'suitcase': [139, 53, 137], 'frisbee': [102, 208, 40], 'skis': [61, 50, 7], 'snowboard': [65, 82, 186],
    'sports ball': [65, 82, 186], 'kite': [153, 254, 81], 'baseball bat': [233, 80, 195],
    'baseball glove': [165, 179, 213], 'skateboard': [57, 65, 211], 'surfboard': [98, 255, 164],
    'tennis racket': [205, 219, 146], 'bottle': [140, 138, 172], 'wine glass': [23, 53, 119], 'cup': [102, 215, 88],
    'fork': [198, 204, 245], 'knife': [183, 132, 233], 'spoon': [14, 87, 125], 'bowl': [221, 43, 104],
    'banana': [181, 215, 6], 'apple': [16, 139, 183], 'sandwich': [150, 136, 166], 'orange': [219, 144, 1],
    'broccoli': [123, 226, 195], 'carrot': [230, 45, 209], 'hot dog': [252, 215, 56], 'pizza': [234, 170, 131],
    'donut': [36, 208, 234], 'cake': [19, 24, 2], 'chair': [115, 184, 234], 'sofa': [125, 238, 12],
    'pottedplant': [57, 226, 76], 'bed': [77, 31, 134], 'diningtable': [208, 202, 204], 'toilet': [208, 202, 204],
    'tvmonitor': [208, 202, 204], 'laptop': [159, 149, 163], 'mouse': [148, 148, 87], 'remote': [171, 107, 183],
    'keyboard': [33, 154, 135], 'cell phone': [206, 209, 108], 'microwave': [206, 209, 108], 'oven': [97, 246, 15],
    'toaster': [147, 140, 184], 'sink': [157, 58, 24], 'refrigerator': [117, 145, 137], 'book': [155, 129, 244],
    'clock': [53, 61, 6], 'vase': [145, 75, 152], 'scissors': [8, 140, 38], 'teddy bear': [37, 61, 220],
    'hair drier': [129, 12, 229], 'toothbrush': [11, 126, 158]
    }

class_names_default = list(color_dict_default)  # extract keys from dictionary

# Set up color_dict and class_names to be detected
# color_dict = color_dict_default
color_dict = {'person': [0, 128, 0]}  # dotect only class 'person'
class_names = class_names_default


class Rectangle:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


def load_net(config_path="./cfg/yolov4.cfg", weight_path="./yolov4.weights"):
    # Checks if file exists otherwise return ValueError
    if not os.path.exists(config_path):
        raise ValueError("Invalid yolo config path: " + os.path.abspath(config_path))
    if not os.path.exists(weight_path):
        raise ValueError("Invalid yolo weight path: " + os.path.abspath(weight_path))
    return darknet.load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)  # batch size=1


def is_overlapping(self: Rectangle, other: Rectangle):
    if self.y_min > other.y_max or self.x_min > other.x_max or self.y_max < other.y_min or self.x_max < other.x_min:
        return False
    return True


def draw_boxes(detections, img):
    people_counter = 0
    people_list = []

    for label, confidence, bbox in detections:
        print(label, confidence)

        if label in color_dict:
            color = color_dict[label]
            x_min, y_min, x_max, y_max = darknet.bbox2points(bbox)
            rect = Rectangle(x_min, y_min, x_max, y_max)
            cv2.rectangle(img, (rect.x_min, rect.y_min), (rect.x_max, rect.y_max), color, 2)
            cv2.putText(img, label + " " + confidence, (rect.x_min, rect.y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # if label is 'person' append to people_list
            if label is 'person':
                people_list.append(rect)
                people_counter += 1

    # detect overlapping
    is_crowd_detected = False
    if len(people_list) > 1:  # check if list is greater than 1
        for index in range(len(people_list) - 1):
            rect_self = people_list[index]
            for index_other in range(index + 1, len(people_list), 1):
                rect_other = people_list[index_other]
                if is_overlapping(rect_self, rect_other) is True:
                    print("Crowd detected")
                    is_crowd_detected = True
                    cv2.rectangle(img, (rect_self.x_min, rect_self.y_min), (rect_self.x_max, rect_self.y_max), (255, 0, 0), 3)
                    cv2.rectangle(img, (rect_other.x_min, rect_other.y_min), (rect_other.x_max, rect_other.y_max), (255, 0, 0), 3)

    # add number of people below image
    text_field = np.full((80, img.shape[1], 3), 235, dtype=img.dtype.name)  # add white board below image to write a text into
    img = np.append(img, text_field, axis=0)
    cv2.putText(img, 'Number of people: ' + str(people_counter), (10, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,0], 1)
    if is_crowd_detected is True:
        cv2.putText(img, 'Crowd detected!', (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,0,0], 2)

    print('Number of people: ', people_counter)
    return img


def run():
    network = load_net()

    # cap = cv2.VideoCapture(0)  # use Webcam

    # TODO check if video path exist
    cap = cv2.VideoCapture("data/ShopAssistant1cor.mpg")  # set path to input video
    if not cap.isOpened():  # check if video is open
        raise SystemExit("Could not open video")

    frame_width = int(cap.get(3))  # returns the width of capture video
    frame_height = int(cap.get(4))  # returns the height of capture video

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3)  # Create image according darknet for compatibility of network

    counter = 1  # counter to numerate images
    start_time = time.time()
    while True:  # load the input frame and write output frame.
        prev_time = time.time()
        success, frame = cap.read()  # Capture frame and return true if frame present. frame is numpy array
        # For Assertion Failed Error in OpenCV
        if not success:  # Check if frame present otherwise break the while loop
            break

        # Opencv uses BRG. Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        # Darknet doesn't accept numpy images. Copy frame bytes to darknet_image
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45)
        # Returns a list (label, confidence, bbox(x, y, w, h))
        # where x, y is the centroid of bounding box, w width, h height
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)

        image = draw_boxes(detections, frame_resized)  # draw colored rectangles around each detected bounding box class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))

        # write image
        cv2.imwrite("results/output/" + str(counter) + ".jpg", image)
        counter += 1

        cv2.imshow('Demo', image)  # Display Image window
        cv2.waitKey(3)

    stop_time = time.time()
    running_time = stop_time - start_time
    fps = counter / running_time
    print("Running time: ", str(running_time), "fps: ", fps)
    cap.release()  # release cap


if __name__ == "__main__":
    run()  # Calls the main function run()
