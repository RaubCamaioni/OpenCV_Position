from matplotlib import pyplot as plt
import numpy as np
import math
import cv2


def plot_translation(translations):
    fig = plt.figure()

    # data
    translations = np.array(translations)
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal", adjustable="box")
    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
    )

    # labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Plot of Translations")

    # same aspect ratio for all axis
    max_val = np.max(np.abs(translations))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    plt.show()


def FrameGenerator(*args, **kwargs):
    """
    opencv Video Capture does not have a built in iterator or context.
    this function ensures the cap is released and makes an iterable.
    """
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        # NOTE: if q is pressed release is called at cleanup
        # a context should be used for more professional code
        cap.release()


def symbol_h_points(h1, h2, h3, h4, w1, w2):
    """
    creats the h symbol from variables
    see symbol_dimensions image for more details

    If you print out the symbol yourself, ensure the measurements are the same.
    A4 paper was used.
    """
    translation_between_points = [
        [0, 0],
        [w1, 0],
        [0, h2],
        [w2, 0],
        [0, -h2],
        [w1, 0],
        [0, h1],
        [-w1, 0],
        [0, -h3],
        [-w2, 0],
        [0, h3],
        [-w1, 0],
    ]
    return np.cumsum(translation_between_points, axis=0)


def right_angles(approx, thresh):
    """summation of internal angle differences to 90 degrees"""

    right = True
    error = 0
    AL = len(approx)
    for i in range(0, AL):
        x1 = approx[i % AL][0][0]
        y1 = approx[i % AL][0][1]
        x2 = approx[(i + 1) % AL][0][0]
        y2 = approx[(i + 1) % AL][0][1]
        x3 = approx[(i + 2) % AL][0][0]
        y3 = approx[(i + 2) % AL][0][1]

        l1 = np.array([x2 - x1, y2 - y1])
        l2 = np.array([x3 - x2, y3 - y2])

        dot = np.dot(l1, l2)
        magnitude = np.linalg.norm(l1) * np.linalg.norm(l2)
        normalied_dot = np.clip(abs(dot) / magnitude, -1, 1)
        angle = np.rad2deg(np.arccos(normalied_dot))

        dif = abs(angle - 90)
        error += dif

        if dif > thresh:
            right = False

    return (right, error / AL)


def draw_circles(image, points, radius, color, thick):
    """draw list of points on image"""
    for i in range(0, len(points)):
        cv2.circle(image, tuple(map(tuple, points))[i], radius, color, thick)
    return 1


def detect_color(frame, color):
    """thresholds given color in frame, centered HUE to remove wrapping errors"""
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    h, s, v = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
    frame[:, :, 0] = (frame[:, :, 0] + 270 - h) % 180
    lower = np.array([80, s - 50, v - 50])
    upper = np.array([90, s + 50, v + 50])
    mask = cv2.inRange(frame.astype(np.uint8), lower, upper)
    return mask


def is_clockwise(points):
    """determines if given points are in clockwise or counter-clockwise oder
    Shoelace Formula: https://en.wikipedia.org/wiki/Shoelace_formula
    """
    area = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[i][1] * points[j][0]
    return area > 0


def sort_contour(circle, pts_dst):
    """Returns pts_dst with clockwise ordering and point closest to circle as first point"""

    circle = np.array(circle)
    pts_dst = np.array(pts_dst)

    # order points clockwise
    clockwise = int(is_clockwise(pts_dst) * 2 - 1)
    pts_dst = pts_dst[::clockwise]

    # make closest point first point
    index = np.argmin(np.linalg.norm(pts_dst - circle, axis=-1))
    pts_dst = np.vstack((pts_dst[index:], pts_dst[:index]))

    return pts_dst


def resize_image(image, max_dimension):
    height, width = image.shape[:2]
    max_dim = max(height, width)
    scale = min(max_dimension / max_dim, 1)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return scale, image


def camera_matrix(fx, fy, width, height):
    cx = width / 2
    cy = height / 2

    K = np.matrix(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )

    return K


# This decomposes the homogrpahy matrix
# Takes two inputs
# A - intrinsic camera matrix
# H - homography between two 2D points


def decompose_homography(A, H):
    H = np.transpose(H)
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]

    Ainv = np.linalg.inv(A)

    L = 1 / np.linalg.norm(np.dot(Ainv, h1))

    r1 = L * np.dot(Ainv, h1)
    r2 = L * np.dot(Ainv, h2)
    r3 = np.cross(r1, r2)

    T = L * np.dot(Ainv, h3)

    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    U, S, V = np.linalg.svd(R, full_matrices=True)

    U = np.matrix(U)
    V = np.matrix(V)
    R = U * V

    return (R, T)


def decompose_rotation(R):
    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2]))
    z = math.atan2(R[1, 0], R[0, 0])
    return (x, y, z)
