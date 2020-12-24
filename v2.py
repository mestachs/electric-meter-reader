import cv2
import pytesseract
import random
import numpy as np
import math


import math

import cv2

from math import pi
import matplotlib.pyplot as plt

debug = False


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def stack_images(img_a, img_b):
    return np.hstack((img_a, img_b))

def show_image(img):
    plt.figure(figsize = (15, 10))
    plt.imshow(img)

def show_gray_image(img):
    plt.figure(figsize = (15, 10))
    plt.imshow(img, cmap='gray')

def plot_start():
    plt.figure(figsize = (15, 10))

def plot_lines(lines, color):
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot((x1, x2), (y1, y2), color)

def plot_function(f, xs):
    for x in range(xs[0], xs[len(xs) - 1]):
        plt.plot(x, f(x), 'g2')

def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def clache(img):
    cl = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y_clache = cl.apply(y)
    img_yuv = cv2.merge((y_clache, u, v))

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def biliteral(img):
    return cv2.bilateralFilter(img, 13, 75, 75)

def unsharp_mask(image, blured):
    return cv2.addWeighted(blured, 1.5, blured, -0.5, 0, image)

def binary_hsv_mask(img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])

    return cv2.inRange(img, lower, upper)

def binary_gray_mask(img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])

    return cv2.inRange(img, color_range[0][0], color_range[1][0])

def binary_mask_apply(img, binary_mask):
    masked_image = np.zeros_like(img)

    for i in range(3):
        masked_image[:,:,i] = binary_mask.copy()

    return masked_image

def binary_mask_apply_color(img, binary_mask):
    return cv2.bitwise_and(img, img, mask = binary_mask)

def filter_by_color_ranges(img, color_ranges):
    result = np.zeros_like(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for color_range in color_ranges:
        color_bottom = color_range[0]
        color_top = color_range[1]

        if color_bottom[0] == color_bottom[1] == color_bottom[2] and color_top[0] == color_top[1] == color_top[2]:
            mask = binary_gray_mask(gray_img, color_range)
        else:
            mask = binary_hsv_mask(hsv_img, color_range)

        masked_img = binary_mask_apply(img, mask)
        result = cv2.addWeighted(masked_img, 1.0, result, 1.0, 0.0)

    return result

def color_threshold(img, white_value):
    white = [[white_value, white_value, white_value], [255, 255, 255]]
    yellow = [[80, 90, 90], [120, 255, 255]]

    return filter_by_color_ranges(img, [white, yellow])



def as_edges(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray_img)
    sigma = 0.33
    lower = int(max(150, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return canny(gray_img, lower, upper)



def hough_lines(image, white_value):

    image_masked = color_threshold(image, white_value)
    image_edges = as_edges(image_masked)
    houghed_lns = cv2.HoughLinesP(image_edges, 2, np.pi / 180, 50, np.array([]), 20, 100)

    if houghed_lns is None:
        return hough_lines(image, white_value - 5)

    return houghed_lns



def is_overlapping_horizontally(box1, box2):
    x1, _, w1, _ = box1
    x2, _, _, _ = box2
    if x1 > x2:
        return is_overlapping_horizontally(box2, box1)
    return (x2 - x1) < w1

def merge(box1, box2):
    assert is_overlapping_horizontally(box1, box2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    w = max(x1 + w1, x2 + w2) - x
    y = min(y1, y2)
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)

def windows(contours):
    """return List[Tuple[x: Int, y: Int, w: Int, h: Int]]"""
    boxes = []
    for cont in contours:
        box = cv2.boundingRect(cont)
        if not boxes:
            boxes.append(box)
        else:
            if is_overlapping_horizontally(boxes[-1], box):
                last_box = boxes.pop()
                merged_box = merge(box, last_box)
                boxes.append(merged_box)
            else:
                boxes.append(box)
    return boxes

def process(filename):
    original = cv2.imread(filename)
    gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    houghed_lns = hough_lines(original,180)
    print(houghed_lns)

    # gray_image =  color_threshold(original,30)
    cv2.imshow('gray_image', gray_image)
	# compute the median of the single channel pixel intensities
    v = np.median(gray_image)
    sigma = 0.01
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray_image, 150, 170, apertureSize=3)
    # edges = cv2.Canny(gray_image, lower, upper, apertureSize=3)
    lines = cv2.HoughLinesP(gray_image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=250)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_dict = dict()
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = cv2.contourArea(cont)
        if area > 5 and w > 5 and h > 10:
            contours_dict[(x, y, w, h)] = cont

    contours_filtered = sorted(contours_dict.values(), key=cv2.boundingRect)

    blank_background = np.zeros_like(edges)
    img_contours = cv2.drawContours(blank_background, contours_filtered, -1, (255,255,255), thickness=10)
    #img_contours = cv2.drawContours(blank_background, contours_filtered, -1, (255,255,255), thickness=cv2.FILLED)
    if debug:
        cv2.imshow('gray', img_contours)

    boxes = windows(contours_filtered)

    img = cv2.imread(filename)
    for box in boxes:
        x, y, w, h = box
        img = cv2.rectangle(img, (x, y), (x + w , y + h ), (0, 255, 0), 2)

    #for line in lines:
    #    x1, y1, x2, y2 = line[0]
    #    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    if not debug:
        cv2.imshow("boxes", img)
        cv2.waitKey(6000)

    return boxes




process('data/Selection_764.png')
process('data/Selection_765_1.png')
process('data/Selection_765_2.png')
process('data/Selection_765_3.png')
