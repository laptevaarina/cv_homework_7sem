import cv2
import numpy as np

from numpy.linalg import lstsq
from scipy.spatial import distance

def houghScoring(ground_image,line, alpha = 1.0):
    if np.median(ground_image[:10][:]) == 1:
        ground_image = (ground_image - 1)

    coordinates = calcLineParams(line, ground_image.shape)
    hough_image = np.zeros_like(ground_image)
    pts = np.array([[0,0],[ground_image.shape[1],0],(coordinates[2],coordinates[3]),(coordinates[0],coordinates[1])])
    _=cv2.drawContours(hough_image, np.int32([pts]),0, 1, -1)

    hough_image = (hough_image - 1)

    cost = calcCost(hough_image, ground_image, alpha)
    return coordinates, cost


def calcLineParams(line, image_shape):
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = 0
    y1 = int(y0 + image_shape[0]*(a))
    x2 = image_shape[1]
    y2 = int(y0 - image_shape[0]*(a))
    return [x1, y1, x2, y2]


def calcCost(hough_image, ground_image, alpha):
    cost = np.sum(np.where(np.logical_xor(hough_image==0, ground_image==0), 1, 0))
    return cost


def calcEuclidianDistance(true_coords,pred_coords):
    eu_dist = np.linalg.norm(np.subtract(true_coords, pred_coords))
    return eu_dist


def getContourLine(img):
    img_gauss = cv2.GaussianBlur(img, (15, 15), 9)

    thr, img_otsu = cv2.threshold(img_gauss[:, :, 0], thresh=0, maxval=1,
                                  type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    land_mask = img_otsu - 1

    opening = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, (21, 21))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (21, 21))

    image_canny_closing = cv2.Canny(image=land_mask, threshold1=100, threshold2=200)

    contours, hierarchy = cv2.findContours(image_canny_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    arclen = cv2.arcLength(contour, True)
    cnt = cv2.approxPolyDP(contour, 0.002 * arclen, False)
    img_cont = np.zeros_like(image_canny_closing)
    img_cont = cv2.drawContours(img_cont, [cnt], 0, 255, 2)
    return img_cont, land_mask


def getHorizonLineCoords(img_contours, land_mask):
    lines = cv2.HoughLines(img_contours, 75, np.pi / 180, 3)
    all_coordinates = []
    all_cost = []

    for i in range(2):
        coordinates, cost = houghScoring(land_mask, lines[i], alpha=1.0)
        all_coordinates.append(coordinates)
        all_cost.append(cost)

    x1, y1, x2, y2 = all_coordinates[np.argmin(all_cost)]
    return [x1, y1, x2, y2]


def detectHorizon(img):
    img_contours, land_mask = getContourLine(img)
    horizon_line_xy = getHorizonLineCoords(img_contours, land_mask)

    return horizon_line_xy

def get_video_details(cap):
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cnt, w, h, fps


def mean_dist(pt, pts):
    md = 0
    for p in pts:
        dst = distance.euclidean(p, pt)
        md += dst

    return md / (len(pts))

def sparse_motion(path):
    color = (0, 255, 0)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=30, qualityLevel=0.2, minDistance=70, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(path)
    cnt, W, H, fps = get_video_details(cap)
    cv2.startWindowThread()
    cv2.namedWindow('imageWindow')
    is_ok, first_frame = cap.read()

    out = cv2.VideoWriter("result1.mp4",
                          -1, 20.0, (int(W), int(H)), isColor=True)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    mask = np.zeros_like(first_frame)
    idx = 0
    while (cap.isOpened()):
        is_ok, frame = cap.read()
        if not is_ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)

        find_mask = np.zeros(frame.shape[:2])

        line = detectHorizon(frame)
        start = (line[0], line[1])
        end = (line[2], line[3])
        pts = [start, end]

        x_coords, y_coords = zip(*pts)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]

        for i in range(int(W)):
            for j in range(int(H)):
                if j < m * i + c:
                    find_mask[j, i] = 1

        prev = cv2.goodFeaturesToTrack(prev_gray, mask=find_mask.astype(np.uint8), **feature_params)

        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

        good_old = prev[status == 1].astype(int)
        good_new = next[status == 1].astype(int)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            dst = mean_dist((a, b), good_new)
            new_frame = cv2.rectangle(frame, (a-5, b-5), (a+5, b+5), color, 2)

        output = cv2.add(frame, mask)
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)

        out.write(output)
        # Opens a new window and displays the output frame
        cv2.imshow("sparse optical flow", output)
        cv2.imwrite(f'tmp/frame_{str(idx).zfill(5)}.jpg', output)
        idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "../videos/Videos/Clip_4.mov"
    sparse_motion(path)

