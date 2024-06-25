import cv2
import numpy as np

def new_borders(M, h: int, w: int):
    """ Находим крайние точки после преобразования и из них считаем новые высоту и ширину картинки"""
    left_upper = M @ np.array([0, 0, 1])
    right_upper = M @ np.array([0, h, 1])

    left_lower = M @ np.array([w, 0, 1])
    right_lower = M @ np.array([w, h, 1])

    low_w = np.min([left_lower[0], left_upper[0], right_lower[0], right_upper[0]])
    high_w = np.max([left_lower[0], left_upper[0], right_lower[0], right_upper[0]])
    new_w = int(np.round(high_w - low_w))

    low_h = np.min([left_lower[1], left_upper[1], right_lower[1], right_upper[1]])
    high_h = np.max([left_lower[1], left_upper[1], right_lower[1], right_upper[1]])
    new_h = int(np.round(high_h - low_h))

    return (low_w, low_h), (new_w, new_h)

def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    h, w, _ = image.shape

    M_rotate = cv2.getRotationMatrix2D(point, angle, scale=1.0)

    low, new_shp = new_borders(M_rotate, h, w)
    M_rotate[0][-1] -= low[0]
    M_rotate[1][-1] -= low[1]

    dst_rotate = cv2.warpAffine(image, M_rotate, (new_shp[0], new_shp[1]))
    return dst_rotate


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    h, w, _ = image.shape

    M = cv2.getAffineTransform(points1, points2)

    low, new_shp = new_borders(M, h, w)
    M[0][-1] -= low[0]
    M[1][-1] -= low[1]

    dst = cv2.warpAffine(image, M, (new_shp[0], new_shp[1]))
    return dst
