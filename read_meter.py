import cv2
import pytesseract
import random
import numpy as np
import math


import math

import cv2

from math import pi
import configparser

class ImageProcessor:

    def __init__(self, img):
        self.img = img
        # https://stackoverflow.com/questions/21483301/how-to-unpack-optional-items-from-a-tuple
        self.height, self.width, _ = (list(self.img.shape) + [None]*3)[:3]

    def process(self):
        """ Do processing here and return the result"""
        return self.img


class Straighten(ImageProcessor):

    def __init__(self, img, debug: bool = False):
        super().__init__(img)
        """

        @type debug: bool
        """
        self.debug = debug
        print('debug', self.debug)

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def process(self):
        print('HoughLines...')
        cThreshold = 140
        print('cThreshold', cThreshold)
        #                                pixel  degree=1        min lines
        lines = cv2.HoughLines(self.img, rho=1, theta=pi / 180,
                               threshold=cThreshold, lines=60 * pi / 180, srn=120 * pi / 180)
        print(len(lines), 'lines')

        if self.debug:
            imageWithLines = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
                cv2.line(imageWithLines, pt1, pt2, (255, 0, 0))
            cv2.imwrite('3-lines.png', imageWithLines)

        # print(lines)
        if lines is not None:
            skew = self.detect_skew(lines)
            print("detectSkew: %.1f deg", skew)

            straight = self.rotate(self.img, -skew)
            return straight
        else:
            return self.img

    def detect_skew(self, lines):
        theta_avr = 0
        for line in lines:
            # print(line)
            theta_avr += line[0][1]

        theta_deg = 0
        if len(lines):
            theta_avr /= len(lines)
            theta_deg = (theta_avr / pi * 180) - 90

        return theta_deg

    def rotate(self, img, skew):
        height, width = img.shape
        M = cv2.getRotationMatrix2D((width / 2, height / 2), skew * 2, 1)
        img_rotated = cv2.warpAffine(img, M, img.shape[::-1])
        return img_rotated

class Cannify(ImageProcessor):

    def __init__(self, img, debug: bool = False):
        super().__init__(img)
        # self.low_area = 300
        # self.high_area = 1100
        self.low_height = 45
        self.high_height = 60
        self.digits = []
        self.debug: bool = debug

        config = configparser.ConfigParser()
        config.read('config.ini')
        self.config: map = {'low_height': '40', 'high_height': '130', 'min_aspect': '0.4', 'max_aspect': '0.6'}
        self.low_height: int = int(self.config['low_height'])
        self.high_height: int = int(self.config['high_height'])
        self.min_aspect: float = float(self.config['min_aspect'])
        self.max_aspect: float = float(self.config['max_aspect'])

    def process(self):
        """
        We are finding contours in the image,
        then we filter the contours that would fit the specified bounds (see config.ini)
        @return:
        """
        cv2.imwrite('5-cannify.png', self.img)
        contours, hierarchy = cv2.findContours(self.img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        print('len contours', len(contours))

        contimage = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.drawContours(contimage, contours, contourIdx=-1, color=(50, 50, 50))
        if self.debug:
            cv2.imwrite('6-contours.png', contimage)

        # contours1 = self.filter_contours_by_area(contours, self.low_area, self.high_area)
        # print('len contours1', len(contours1))
        # cv2.drawContours(contimage, contours1, contourIdx=-1, color=(255, 255, 0))

        contours2 = self.filter_contours_by_height(contours, self.low_height, self.high_height)
        print('len contours2', len(contours2))
        cv2.drawContours(contimage, contours2, contourIdx=-1, color=(0, 255, 0))
        if self.debug:
            cv2.imwrite('6-contours2.png', contimage)

        contours3 = self.filter_contours_by_aspect(contours2, self.min_aspect, self.max_aspect)
        print('len contours3', len(contours3))
        cv2.drawContours(contimage, contours3, contourIdx=-1, color=(0, 0, 255))
        if self.debug:
            cv2.imwrite('6-contours3.png', contimage)

        average_y, average_height = self.get_average_height(contours3)
        if average_height is not None and average_height > 0:
            average_height *= 1.2
            print('average_y', average_y)
            print('average_height', average_height)
            if average_y and average_height:
                cv2.line(contimage, (0, math.floor(average_y - 0)),
                         (self.width, math.floor(average_y - 0)), color=(255, 0, 0))
                cv2.line(contimage, (0, math.floor(average_y - average_height)),
                         (self.width, math.floor(average_y - average_height)), color=(0, 0, 255))
                cv2.line(contimage, (0, math.floor(average_y + average_height)),
                         (self.width, math.floor(average_y + average_height)), color=(0, 0, 255))
                contours4 = self.filter_contours_by_position(contours3, average_y, average_height)
                print('len contours4', len(contours4))
                cv2.drawContours(contimage, contours4, contourIdx=-1, color=(0, 255, 255))
                if self.debug:
                    cv2.imwrite('6-contours4.png', contimage)

                contours5 = self.reintroduce_inner_elements(contours4, contours)
                cv2.drawContours(contimage, contours5, contourIdx=-1, color=(255, 0, 255))
                if self.debug:
                    cv2.imwrite('6-contours5.png', contimage)

                self.digits = contours5

        return contimage

    def click(self):
        # self.low_height += 5
        print(self.low_area, 'area', self.high_area)
        print(self.low_height, 'height', self.high_height)

    def filter_contours_by_area(self, contours, min_area, max_area):
        good = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                good.append(c)

        return good

    def filter_contours_by_height(self, contours, min_height, max_height):
        good = []
        index = 1
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            print("contour",x, y, w, h)


            #if min_height <= h <= max_height and w > 10:
                # print('h', h)
            good.append(c)
            index = index +1
        return good

    def filter_contours_by_aspect(self, contours, desired_aspect, sigma):
        good = []
        aspect_list = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            # print('a', aspect_ratio)
            aspect_list.append(aspect_ratio)
            if np.isclose(aspect_ratio, desired_aspect, sigma):
                good.append(c)

        print('average aspect', self.mean(aspect_list))
        return good

    def mean(self, numbers):
        return float(sum(numbers)) / max(len(numbers), 1)

    def get_average_height(self, contours):
        y_list = []
        height_list = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            y_list.append(y)
            height_list.append(h)

        average_y = None
        average_height = None
        # average_y = self.mean(y_list)
        if len(y_list):
            average_y = max(y_list, key=y_list.count)
            average_height = self.mean(height_list)
        return average_y, average_height

    def filter_contours_by_position(self, contours, average_y, average_height):
        good = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ok = (average_y - average_height) <= y <= (average_y + average_height)
            # print((average_y - average_height), y, (average_y + average_height), ok)
            if ok:
                good.append(c)

        return good

    def reintroduce_inner_elements(self, contours_big, contours_all):
        small = []
        for n in contours_big:
            nx, ny, nw, nh = cv2.boundingRect(n)
            for c in contours_all:
                x, y, w, h = cv2.boundingRect(c)
                if w <= nw and h <= nh and x >= nx and y >= ny and x <= (nx + nw) and y <= (ny + nh):
                    small.append(c)

        print('small', len(small))
        return contours_big + small

    def getDigits(self):
        hashes = []
        unique = []
        for c in self.digits:
            h = hash(c.tobytes())
            if h not in hashes:
                hashes.append(h)
                unique.append(c)

        print('unique', len(self.digits), len(unique))
        return unique



class IsolateDigits(ImageProcessor):
    """
    We could use findContours() on the image again, but we have contours already
    Give up
    https://stackoverflow.com/questions/29523177/opencv-merging-overlapping-rectangles
    """

    def __init__(self, img):
        super().__init__(img)

    def isolate(self, contours):
        digits = []
        # must be RETR_EXTERNAL this time
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        # remove tiny specs which happen to fit bounding box of big characters
        contimage = np.zeros((self.height, self.width, 3), np.uint8)
        cannify2 = Cannify(contimage)
        contours = cannify2.filter_contours_by_height(contours, cannify2.low_height, cannify2.high_height)

        # sort by x
        contours = sorted(contours, key=self.sort_by_x)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            digits.append(self.img[y:y+h, x:x+w])

        return digits

    @staticmethod
    def sort_by_x(c):
        x, y, w, h = cv2.boundingRect(c)
        return x

    def isolate_by_contours(self, contours):
        to_delete = []
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            b1 = [x, y, x+w, y+h]
            for ni, n in enumerate(contours[i:]):
                nx, ny, nw, nh = cv2.boundingRect(n)
                b2 = [nx, ny, nx+nw, ny+nh]
                intersecting = len(self.intersection(b1, b2))
                if intersecting:
                    to_delete.append(ni)

        c2 = []
        for i, c in enumerate(contours):
            if i not in to_delete:
                c2.append(c)

        print('after isolate', len(contours), len(c2))
        return c2

    def union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0] + a[2], b[0] + b[2]) - x
        h = max(a[1] + a[3], b[1] + b[3]) - y
        return x, y, w, h

    def intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y
        if w < 0 or h < 0: return ()  # or (0,0,0,0) ?
        return x, y, w, h


image = cv2.imread('data/Selection_765.png')#20201219_133828.jpeg')#Selection_765.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#window_name = 'image'
#cv2.imshow(window_name,gray_image)
#cv2.waitKey(0)

edges = cv2.Canny(gray_image, 200, 100, apertureSize=3)
image_ret = edges.copy()
cv2.imshow("grey canny",image_ret)

custom_config = r'--oem 3 --psm 7 outbase digits'
print(pytesseract.image_to_string(image_ret, config=custom_config))

debug = True

straighten = Straighten(edges, debug=debug)
straight = image_ret

cannify = Cannify(straight, debug=debug)
contimage = cannify.process()
contours = cannify.getDigits()

height, width = image_ret.shape
isolated = np.zeros((height, width, 3), np.uint8)
cv2.drawContours(isolated, contours, contourIdx=-1, color=(255, 255, 255))

isolator = IsolateDigits(isolated)
digits = isolator.isolate(contours)


index = 0
for digit in contours:
    try:
        pass
        cv2.imshow("digits-"+str(index),digit)
    except:
        pass
    index = index + 1
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)


custom_config = r'--oem 3 --psm 7 outbase digits'
print(pytesseract.image_to_string(digits[0], config=custom_config))

cv2.destroyAllWindows()
