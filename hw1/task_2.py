import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """

    yellow = {0: range(240, 255), 1: range(240, 255), 2: range(110, 165)}
    grey = {0: range(180, 220), 1: range(180, 220), 2: range(180, 220)}

    # считаем количество дорожных полос на картинке
    count_lanes = -1
    flag = 0
    for clr in image[0]:
        if (clr[0] in yellow[0]) and (clr[1] in yellow[1]) and (clr[2] in yellow[2]) and flag == 0:
            flag = 1
            count_lanes += 1
        elif (clr[0] in grey[0]) and (clr[1] in grey[1]) and (clr[2] in grey[2]):
            flag = 0

    lower_blue = np.array([40, 100, 240])
    upper_blue = np.array([60, 140, 255])

    lower_red = np.array([240, 0, 0])
    upper_red = np.array([255, 50, 50])

    # находим номер полосы, на которой находится машина
    mask_blue = cv2.inRange(image, lower_blue, upper_blue)
    bbox_blue = cv2.boundingRect(mask_blue)

    shp = image.shape
    road_width = shp[1] // count_lanes
    road_number = 0
    for i in range(count_lanes):
        coordx_range = range(i * road_width, (i + 1) * road_width)
        if bbox_blue[0] in coordx_range:
            road_number = i
            break

    # находим номер свободной полосы
    free_road = 0
    for i in range(count_lanes):
        road = image[:, i * road_width:(i + 1) * road_width]
        mask_brick = cv2.inRange(road, lower_red, upper_red)
        bbox_red = cv2.boundingRect(mask_brick)
        if bbox_red == (0, 0, 0, 0):
            free_road = i
            break

    # if road_number == free_road:
    #     print("Перестраиваться не нужно\n")

    return free_road
