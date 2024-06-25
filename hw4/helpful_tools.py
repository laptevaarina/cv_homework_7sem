import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
import random
from scipy.ndimage import affine_transform


class Object:
    def __init__(self, centre, length, width):
        self.centre = centre
        self.length = length
        self.width = width

# plt.style.use('seaborn')
# %matplotlib inline


### Вспомогательные функции

def plot_transform_result(src_image, transform_image, is_gray=False):
    """
    Отрисовать с помощью plt исходное изображение и его преобразование.

    :param src_image: np.ndarray: исходное изображение
    :param transform_image: np.ndarray: преобразованное изображение
    :param is_gray: bool: флаг для отображения ЧБ изображений
    :return: None
    """
    fig, m_axs = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8 * 2), constrained_layout=True)
    ax1, ax2 = m_axs

    cmap = 'gray' if is_gray else None
    ax1.set_title('Исходное изображение')
    ax1.imshow(src_image, cmap=cmap)
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_title('Результат преобразования')
    ax2.imshow(transform_image, cmap=cmap)
    ax2.set_xticks([]), ax2.set_yticks([])
    plt.show()


def plot_one_image(src_image, is_gray=False):
    """
    Отрисовать с помощью plt исходное изображение.

    :param src_image: np.ndarray: исходное изображение
    :param is_gray: bool: флаг для отображения ЧБ изображений
    :return: None
    """
    fig, m_axs = plt.subplots(1, 1, figsize=(6.4 * 2, 4.8 * 2), constrained_layout=True)
    ax1 = m_axs

    cmap = 'gray' if is_gray else None
    ax1.set_title('Исходное изображение')
    ax1.imshow(src_image, cmap=cmap)
    ax1.set_xticks([]), ax1.set_yticks([])
    plt.show()



### Спектральный анализ

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def idealFilterLP(D0, img_size):
    base = np.zeros(img_size[:2])
    rows, cols = img_size[:2]
    center = (rows / 2, cols / 2)

    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1

    return base


def idealFilterHP(D0, img_size):
    base = np.ones(img_size[:2])
    rows, cols = img_size[:2]
    center = (rows / 2, cols / 2)

    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 0

    return base


def remove_LF_IF(img, delta, view = False):
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    LowPassCenter = center * idealFilterLP(delta, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_LowPass_abs = np.abs(inverse_LowPass)

    if view:
        plt.imshow(inverse_LowPass_abs)
        plt.show()

    return inverse_LowPass_abs


def remove_HF_IF(img, delta, view=False):
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    LowPassCenter = center * idealFilterHP(delta, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPassCenter)
    inverse_LowPass_abs = np.abs(inverse_LowPass)

    if view:
        plt.imshow(inverse_LowPass_abs)
        plt.show()

    return inverse_LowPass_abs


##  Butterworth Filter



def butterworthLP(D0, img_size, n):
    base = np.zeros(img_size[:2])
    rows, cols = img_size[:2]
    center = (rows / 2, cols / 2)

    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))

    return base


def butterworthHP(D0, img_size, n):
    base = np.zeros(img_size[:2])
    rows, cols = img_size[:2]
    center = (rows / 2, cols / 2)

    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))

    return base


# remove hight freq butterworth filt
def remove_LF_BF(img, delta, blur_bord, view=False):
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)

    LowPassCenter = center * butterworthLP(delta, img.shape, blur_bord)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_LowPass_abs = np.abs(inverse_LowPass)

    if view:
        plt.imshow(inverse_LowPass_abs)
        plt.show()

    return inverse_LowPass_abs


def remove_HF_BF(img, delta, blur_bord, view=False):
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)

    LowPassCenter = center * butterworthHP(delta, img.shape, blur_bord)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_LowPass_abs = np.abs(inverse_LowPass)

    if view:
        plt.imshow(inverse_LowPass_abs)
        plt.show()

    return inverse_LowPass_abs


def gaussianLP(D0, img_size):
    base = np.zeros(img_size[:2])
    rows, cols = img_size[:2]
    center = (rows / 2, cols / 2)

    for x in range(cols):
        for y in range(rows):
            base[y, x] = np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))

    return base


def gaussianHP(D0, img_size):
    base = np.zeros(img_size[:2])
    rows, cols = img_size[:2]
    center = (rows / 2, cols / 2)

    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))

    return base


def remove_LF_GF(img, delta, view=False):
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)

    LowPassCenter = center * gaussianLP(delta, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_LowPass_abs = np.abs(inverse_LowPass)

    if view:
        plt.imshow(inverse_LowPass_abs)
        plt.show()

    return inverse_LowPass_abs


def remove_HF_GF(img, delta, view=False):
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)

    LowPassCenter = center * gaussianHP(delta, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    inverse_LowPass_abs = np.abs(inverse_LowPass)

    if view:
        plt.imshow(inverse_LowPass_abs)
        plt.show()

    return inverse_LowPass_abs



### Морфологические трансформации

def dilation(img, kernel_h, kernel_w, view=False):
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)

    if view:
        plt.imshow(dilation)
        plt.show()

    return dilation


def erosion(img, kernel_h, kernel_w, view=False):
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    if view:
        plt.imshow(erosion)
        plt.show()

    return erosion


def opening(img, kernel_h, kernel_w, view=False):
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    if view:
        plt.imshow(opening)
        plt.show()

    return opening


def closing(img, kernel_h, kernel_w, view=False):
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if view:
        plt.imshow(closing)
        plt.show()

    return closing


def gradient(img, kernel_h, kernel_w, view=False):
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    if view:
        plt.imshow(gradient)
        plt.show()

    return gradient



### Поиск границ

def get_border(img, sigma, view=False):
    edges = feature.canny(img, sigma)
    edges = edges.astype(float)

    if view:
        plt.imshow(edges)
        plt.show()

    return edges


def get_contur(img, contur_len, scale = 0.15, view=False):
    laplac = cv2.Laplacian(img, cv2.THRESH_BINARY, scale, ksize=5)
    contours, hierarchy = cv2.findContours(laplac, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []

    for cnt in contours:
        if cnt.shape[0] > contur_len:
            new_contours.append(cnt)

    #     Также найдём крайние точки
    # for point in cnt

    img_cont = img

    cv2.drawContours(img_cont, new_contours, -1, (255, 0, 0), 3)


    if view:
        plot_one_image(img_cont)

    return img_cont


def get_contur_aprox(img, contur_len, epsilon_coeff, scale = 0.15, view=False):
    laplac = cv2.Laplacian(img, cv2.THRESH_BINARY, scale, ksize=5)
    contours, hierarchy = cv2.findContours(laplac, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    # max_l = contours[0].shape[0]

    for cnt in contours:
        if cnt.shape[0]>contur_len:
            contur_len = cnt.shape[0]

            epsilon = epsilon_coeff * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            new_contours.append(approx)

    img_cont = img
    cv2.drawContours(img_cont, new_contours, -1, (255, 0, 0), 3)

    if view:
        plot_one_image(img_cont)

    # img_cont = img_cont*255
    return img_cont


# Ищет крайние пиксели объекта
def get_extreme_pxls_line(line):
    first_pxl = -1
    last_pxl = -1
    not_null = []
    for i, pxl in enumerate(line):
        if (pxl == 255):
            if i > last_pxl: last_pxl = i
            if first_pxl == -1: first_pxl = i
    extreme_pxls = [first_pxl, last_pxl]
    return extreme_pxls


def get_not_null_pxls_line(line):
    new_line=[False]
    first_pxl = -1
    last_pxl = -1
    not_null = []
    for i, pxl in enumerate(line):

        if (pxl == 255):
            not_null.append(i)

    return not_null


# Ищет координаты начала(x), конца(x), боковых сторон(y) по отрисованной
# в чб копии нашего объекта
def get_obj(image):
    top_coord = -1
    left_s_coord = -1
    back_coord = -1
    right_s_coord = -1


    for i, line in enumerate(image):

        if (1 in line) or (255 in line):
            if top_coord == -1:
                top_coord = i

            extreme_pxls = get_extreme_pxls_line(line)

            if (left_s_coord == -1) or (extreme_pxls[0] < left_s_coord):
                left_s_coord = extreme_pxls[0]

            if (right_s_coord == -1) or (extreme_pxls[1] > right_s_coord):
                right_s_coord = extreme_pxls[1]

            #             left_s_coord, right_s_coord = extreme_pxls[0], extreme_pxls[1]

            # if (i < len(image)-1) and (not 255 in image[i + 1]):
            back_coord = i

    coords = list([top_coord+1, left_s_coord+1, back_coord+1, right_s_coord+1])


    return coords


# Рисует рамку вокруг объекта, по координатам, которые берём из
# search_extreme_pxls_img(), не меняет самого
def draw_frame(image, obj:Object):

    frame_point = [
        int(obj.centre[0] - obj.length / 2),
        int(obj.centre[0] + obj.length / 2),
        int(obj.centre[1] - obj.width / 2),
        int(obj.centre[1] + obj.width / 2)
    ]

    key_point = [
        int(obj.centre[1] - obj.width / 2),
        int(obj.centre[0] - obj.length / 2),
        int(obj.centre[1] + obj.width / 2),
        int(obj.centre[0] + obj.length / 2),
    ]

    for i in range(frame_point[0], frame_point[1]):
        image[i][frame_point[2]] = 0
        image[i][frame_point[3]] = 0

    for i in range(frame_point[2], frame_point[3]):
        image[frame_point[0]][i] = 0
        image[frame_point[1]][i] = 0

    return key_point



### Получение масок

# Подал строку с контурами - получил закрашенную строку
def get_mask_line(line):

    new_mask = [0]*len(line)

    nn_line = get_not_null_pxls_line(line)

    # new_mask[0: nn_line[0]] = 0

    count_in = 0
    flag = True

    pair = []
    in_out = []

    for i in range(0, len(nn_line)):

        if (nn_line[i] - nn_line[i-1] > 1):
            pair.append(nn_line[i-1])

        if len(pair) == 2:
            in_out.append(pair)
            pair = []

        if (nn_line[i] != nn_line[i-1]+1) or (i==(len(nn_line)-1)) or (i==0):
            pair.append(nn_line[i])

        if len(pair) == 2:
            in_out.append(pair)
            pair = []

    in_out_fin = []
    pair = []

    if len(in_out)%2 == 0:

        for i in range(len(in_out)):
            if i%2 == 0:
                pair.append(in_out[i][0])
            if i%2 == 1:
                pair.append(in_out[i][1])
            if len(pair) == 2:
                in_out_fin.append(pair)
                pair = []

    if len(in_out)%2 == 1:

        for i in range(len(in_out)):
            if i%2 == 0:
                pair.append(in_out[i][0])
            if i%2 == 1:
                pair.append(in_out[i][1])
                in_out_fin.append(pair)
                pair = []
                pair.append(in_out[i][0])

            if len(pair) == 2:
                in_out_fin.append(pair)
                pair = []

    return in_out_fin

def get_mask_contur(contur, delta):
    for str in range(len(contur)):

        mask_line = get_mask_line(contur[str])
        for rng in mask_line:
            contur[str][rng[0]-delta:rng[1]+delta] = 255

            # if str+delta<len(contur):
            contur[str-delta:str+delta][rng[0]:rng[1]] = 255

    return contur


# Функция конкретно чтобы вырезать изображение хвоста из чёрно-белого
# Путём шаманства с подбором коэффициентов находим хорошие параметры
# для обнаружения большей части хвостов
def get_whale_mask(img):
    mod_img = remove_LF_GF(img, 26)
    edges = get_border(mod_img, 15.8)

    contur = get_contur(edges, 850, scale=0.25, view=False)
    contur = get_contur_aprox(edges, 850, 0.0037, scale=0.25, view=False)

    mod_contur = erosion(contur, 2, 2)
    mod_contur = dilation(mod_contur, 30, 53)
    mod_contur = erosion(mod_contur, 30, 22, False)

    img_mask = get_mask_contur(mod_contur, 10)
    img_mask = opening(img_mask, 55, 55, False)

    return img_mask



### Нормировка изображения

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def read_cropped_image(p, mask_h_w, augment):

    img_shape = (384, 384, 1)  # Форма изображения, используемая моделью
    anisotropy = 2.15  # горизонтальная степень сжатия
    crop_margin = 0.05


    # Если указан идентификатор изображения, он преобразуется в имя файла
    size_x, size_y = p.shape

    # Определить область исходного изображения для захвата на основе ограничительной рамки.
    x0, y0, x1, y1 = mask_h_w
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    # if (x0 < 0): x0 = 0
    # if (x1 > size_x): x1 = size_x
    # if (y0 < 0): y0 = 0
    # if (y1 > size_y): y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

        # Генерация матрицы преобразования
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Используйте аффинное преобразование
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = p
    # img = p.reshape(p.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)

    # Нормализовано до нуля среднего и единицы дисперсии
    img -= np.mean(img)
    img /= np.std(img)
    # img = img[10:160, 150:330]
    return img




