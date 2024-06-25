import cv2
import numpy as np


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """

    a = 0  # сторона квадратика обхода
    for i in range(image.shape[1]):
        if (image[0, i] == np.array([255, 255, 255])).all():
            a += 1

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = np.zeros(image.shape)
    cv2.drawContours(draw, contours, 0, (0, 0, 255), 0)

    kernel = np.ones((a, a))
    img_dilation = cv2.dilate(draw, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT)
    diff = cv2.absdiff(img_dilation, img_erosion)

    x = np.where(np.all(diff == np.array([0, 0, 255]), axis=-1))[0]
    y = np.where(np.all(diff == np.array([0, 0, 255]), axis=-1))[1]
    coords = (x, y)

    return coords
