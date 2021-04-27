from ctypes import *
import math
import os
import cv2
import numpy as np
import time
import itertools
import darknet

# CONSTANT VARIABLES

white_board_height = 120  # white board added to the image to write data into it
mscoco_names_color = {  # mscoco dataset class names
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
class_names = list(mscoco_names_color)  # extract keys from dictionary


class Rectangle:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


def check_path_errors(config_path, weights_path):
    """
    Checks if paths to the files are correct. If not, raises ValueError
    :param config_path:
    :param weights_path:
    :return: void
    """
    # Checks if file exists otherwise return ValueError
    if not os.path.exists(config_path):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(config_path))))
    if not os.path.exists(weights_path):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(weights_path))))


def is_overlapping(r1: Rectangle, r2: Rectangle):
    """
    Return True if rectangles are overlapping
    :param r1: Rectangle
    :param r2: Rectangle
    :return: True is rectangles are overlapping
    """
    # check if rectangles are above, below, on the right or on the left of each other
    if r1.y_min > r2.y_max or r1.x_min > r2.x_max or r1.y_max < r2.y_min or r1.x_max < r2.x_min:
        return False
    return True


def x_threshold(y_coordinate):
    """
    Return a threshold for a given y_coordinate.
    :param y_coordinate:
    :return: x threshold which is horizontal minimum distance in pixel
    """
    return math.ceil(0.59684 * y_coordinate + 28.91704)


def y_threshold(y_coordinate):
    """
    Return a threshold for a given y_coordinate.
    :param y_coordinate:
    :return: y threshold which is vertical minimum distance in pixel
    """
    return math.ceil(0.10172 * y_coordinate - 3.67487)


def detect_short_distance(r1: Rectangle, r2: Rectangle):  # Rectangle(x_min, y_min, x_max, y_max)
    """
    Return True if distance between objects is shorter than threshold.
    Distance is considered as shorter than threshold, if distances between two points are below both x and y thresholds.
    A point is a centroid (middle point) of bottom side of the rectangle.
    y_max coordinate of r1 Rectangle is used as y_coordinate parameter for x_threshold and y_threshold functions.
    :param r1: Rectangle
    :param r2: Rectangle
    :return: True if distance between objects is shorter than threshold
    """
    # calculate centroid (middle point) of bottom side of the rectangle
    r1_x_mid = r1.x_min + (r1.x_max - r1.x_min) / 2  # add a half value between min and max to the min.
    r2_x_mid = r2.x_min + (r2.x_max - r2.x_min) / 2
    # calculate threshold based on y_max coordinate of r1 Rectangle
    # TODO use the greater y_max coordinate from r1 and r2. It will give a greater threshold
    r1_x_threshold = x_threshold(r1.y_max)
    r1_y_threshold = y_threshold(r1.y_max)
    return abs(r1_x_mid - r2_x_mid) < r1_x_threshold and abs(r1.y_max - r2.y_max) < r1_y_threshold


def filter_person(detections, confidence_threshold=0.25):
    """
    Returns list of detected objects of class 'person' above given confidence
    :param detections: list of detections made by YOLO detector. Detection is a tuple containing label, confidence, bbox
    :param confidence_threshold: confidence threshold witch default equal 0.25
    :return: list of detected objects class 'person'
    """
    list = []
    for label, confidence, bbox in detections:
        # filter on label and confidence
        if label is 'person' and float(confidence) >= confidence_threshold:
            list.append((label, confidence, bbox))
    return list


def transform_bbox_to_rect(detections):
    """
    Returns a list of tuples with bbox transformed to rectangle.
    :param detections: list of detections made by YOLO detector. Detection is a tuple containing label, confidence, bbox
    :return: list of detected objects with bbox transformed to rectangle
    """
    list =[]
    for label, confidence, bbox in detections:
        # transform bbox to rectangle
        x_min, y_min, x_max, y_max = darknet.bbox2points(bbox)
        rect = Rectangle(x_min, y_min, x_max, y_max)
        list.append((label, confidence, rect))
    return list


def draw_boxes(detections, img, color=[0, 128, 0]):
    """
    Draws green boxes around detected objects and writes text above rectangle
    :param detections: detections with rectangle instead of bbox
    :param img:
    :param color: color of rectangle and text. Default is green.
    """
    for label, confidence, rect in detections:
        cv2.rectangle(img, (rect.x_min, rect.y_min), (rect.x_max, rect.y_max), color, 2)
        cv2.putText(img, label + " " + confidence, (rect.x_min, rect.y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def print_person_data(detections):
    """
    Prints info to the shell
    :param detections:
    """
    print("Number of people: " + str(len(detections)))
    for label, confidence, rect in detections:
        print("Detected " + label + " with confidence " + confidence)


def filter_crowd_combination(detections):
    """
    Return a list of combinations of size 2, for which objects are in close distance
    :param detections: detections with rectangle instead of bbox
    :return: list of lists of length 2
    """
    crowd_combination = []
    # detect based on mapped distance
    combinations = list(map(list, itertools.combinations(detections, 2)))  # combinations of size 2
    for combination in combinations:
        r1 = combination[0][2]  # get rectangle from first item
        r2 = combination[1][2]  # get rectangle from second item
        if detect_short_distance(r1, r2) is True:
            crowd_combination.append(combination)
    return crowd_combination


def merge_combinations_into_groups(crowd_combination):
    """
    Return a list of crowd groups.
    Takes list of crowd combinations and creates a list of crowd groups.
    A crowd combination is added to a crowd group if it has at least one the same object that is in the crowd group.
    :param crowd_combination: list of crowd combination lists
    :return: list of lists
    """
    run_loop = True  # control while loop
    break_all = False  # control outer loop

    if len(crowd_combination) > 1:
        while run_loop and len(crowd_combination) > 1:
            for outer in range(len(crowd_combination) - 1):  # do not check the last list. Last list is compared to this list
                c1_list = crowd_combination[outer]  # take the first crowd list
                for inner in range(outer + 1, len(crowd_combination), 1):  # start from the next list and loop to the last list
                    c2_list = crowd_combination[inner]  # take the next crowd list and check if any item of that list is in the first list
                    # check if any item in c2_list is in c1_list
                    for i in range(len(c2_list)):
                        if c2_list[i] in c1_list:  # if any item from c2 is in c1:
                            c1_list.extend(c2_list)  # merge lists
                            crowd_combination.pop(inner)  # remove c2 from list
                            temp_set = set(c1_list)  # remove duplicates, set removes duplicates
                            crowd_combination[outer] = list(temp_set)  # convert to list and assign
                            break_all = True  # break all loops (besides while) and start from the beginning
                            # adding new item to the c1_list caused that all crowd combination lists must be checked from the beginning
                            break
                    if break_all:
                        break
                if break_all:
                    break_all = False  # set up break_all to False to be reused in next while loop iteration
                    break
                else:
                    if outer == len(crowd_combination) - 2:  # stop while loop if outer for loop finished without break
                        run_loop = False
    return crowd_combination


def write_data_on_white_board(detections, crowd_groups, img):
    """
    Return image with a white board and text written on it with information about 'person' and crow detections.
    It appends a white board to the bottom of an image with height = white_board_height
    :param detections:
    :param crowd_groups:
    :param img:
    :return: image
    """
    # add white board below image to write a text into
    text_field = np.full((white_board_height, img.shape[1], 3), 235, dtype=img.dtype.name)
    img = np.append(img, text_field, axis=0)
    # write number of people below image
    cv2.putText(img, 'Number of people: ' + str(len(detections)), (10, img.shape[0] + 20 - white_board_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,0], 1)
    # write info about crowd
    if len(crowd_groups) > 0:
        cv2.putText(img, 'Number of crowds: ' + str(len(crowd_groups)), (10, img.shape[0] + 40 - white_board_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,0,0], 2)
        for index in range(len(crowd_groups)):
            cv2.putText(img, str(len(crowd_groups[index])) + ' people in crowd no ' + str(index + 1), (10, img.shape[0] + 60 + 15 * index - white_board_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)
    return img


def print_crowd_data(crowd_groups):
    """
    Prints info to the shell
    :param crowd_groups:
    """
    print("Number of crowds: " + str(len(crowd_groups)))
    for index, crowd_group in enumerate(crowd_groups):
        print("Number of people i crowd no " + str(index + 1) + ": " + str(len(crowd_group)))


def draw_data_on_image(detections, img):
    """
    Returns image with data drawn on it, such as rectangles, text, extra data below image.
    Prints data to the shell.
    :param detections: detections made by YOLO
    :param img: image to draw into
    :return: image with data draw on it
    """
    # CLASS 'PERSON' - drawing and writing to the shell data about detected objects class 'person'
    # filer class 'person'
    detections = filter_person(detections, confidence_threshold=0.25)
    # transform bbox to rectangle
    detections = transform_bbox_to_rect(detections)
    # draw green rectangle and write text
    draw_boxes(detections, img)
    # print info to shell
    print_person_data(detections)

    # CROWD DETECTION - drawing and writing to the shell data about detected crowds
    # list of crowds
    crowd_combination = filter_crowd_combination(detections)
    crowd_groups = merge_combinations_into_groups(crowd_combination)
    # draw rectangles around boxes in crowd
    for index, crowd_group in enumerate(crowd_groups):
        draw_boxes(crowd_group, img, color=[255 - 50 * index, 0, 0])  # use different tone of color for each group
    # add white board to the image and write data on white board
    img = write_data_on_white_board(detections, crowd_groups, img)
    # print crowd data to the shell
    print_crowd_data(crowd_groups)

    # return image
    return img


def run(config_path, weights_path, video_path=0):  # use webcam and YOLOv4 as default
    """
    Runs YOLO detector on video and processes images.
    Exit program by pressing 'q' on displayed image.

    OpenCV Source: https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
    :param video_path: 0 is a default for using webcam. Video path for other video source
    :param config_path:
    :param weights_path:
    :return: void
    """
    # load YOLO detector
    network = darknet.load_net_custom(config_path.encode("ascii"), weights_path.encode("ascii"), 0, 1)  # batch size=1
    # load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  # check if video is open
        raise SystemExit("Could not open video source.")
    frame_width = int(cap.get(3))  # returns the width of captured video
    frame_height = int(cap.get(4))  # returns the height of captured video

    # set up output to make video from images
    # add white_board_height to the frame_height if it was used in image processing
    out = cv2.VideoWriter('results/video/output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (frame_width, frame_height + white_board_height))

    # Create a darknet image. It is reused for each while loop iteration
    darknet_image = darknet.make_image(frame_width, frame_height, 3)  # darknet image must be compatible with processed image

    counter = 1  # counter is used as filename for output image
    start_time = time.time()
    while True:  # load the input frame and write output frame.
        success, frame = cap.read()  # Capture frame and return true if frame present. Frame is a numpy array
        if not success:  # Check if frame captured successfully, otherwise break the while loop
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Opencv uses BRG. Convert frame from BGR to RGB
        frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        # Darknet doesn't accept numpy images. Copy frame bytes to darknet_image
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # info about darknet function detect_image:
        # - parameters: (network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45)
        # - returns detections, which is a list of tuples (label, confidence, bbox(x, y, w, h)),
        #   where x, y is the centroid of bounding box, w width, h height
        start_detection = time.time()  # check detection time
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
        print(round(1/(time.time()-start_detection), 2), 'FPS')

        # draw data on image (rectangles, crowds, text, e.c.t)
        image = draw_data_on_image(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # write image to output
        cv2.imwrite("results/output/" + str(counter) + ".jpg", image)
        counter += 1
        # write image into output video
        out.write(image)
        # display image in window
        cv2.imshow('Demo', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # break loop on 'q' pressed
            break

    cap.release()  # release cap
    out.release()  # release video

    running_time = time.time() - start_time
    fps = round(counter / running_time, 2)
    print("Running time: ", str(running_time), "fps: ", fps)


if __name__ == "__main__":
    # video file path
    # video = "data/input_video_8.mpg"  # Video Source: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
    video = "data/input_video_2.mpg"  # Video Source: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/

    # YOLO config and weights files path
    # yolo_cfg = "./cfg/yolov3-tiny.cfg"
    # yolo_weights = "./yolov3-tiny.weights"

    # YOLOv4
    yolo_cfg = "./cfg/yolov4.cfg"
    yolo_weights = "./yolov4.weights"

    check_path_errors(config_path=yolo_cfg, weights_path=yolo_weights)
    run(yolo_cfg, yolo_weights, video)  # Calls the main function run()
    # run()  # run webcam as default
