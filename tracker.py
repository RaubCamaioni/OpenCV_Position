import numpy as np
import utility
import cv2


def canny_edge_detection(frame):
    """canny edge detection with pre filters"""

    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    blur_grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(blur_grey, 50, 60)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, (3, 3))
    return canny


def find_shape(frame):
    """find shape with 12 side and approximatly 90 degree internal angles"""

    (cont, _) = cv2.findContours(frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_approx = None
    lowest_error = float("inf")

    for c in cont:
        perim = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perim, True)
        area = cv2.contourArea(c)

        if len(approx) == 12 and area > 200:
            # TODO: Looking back, I am not confident in this filtering method.
            # I remember doing a lot of tuning, probably works ok for the 'H' symbol.
            # A more general algorithm would be an interesting research topic, or might already exist?

            # Current algorithm only works for shapes with 90 degree internal angles
            # Algorithm stops working at sharp view angles
            right, error = utility.right_angles(approx, 40)
            if error < lowest_error and right:
                lowest_error = error
                best_approx = approx

    return best_approx


def track_frame(A, frame, points_2d, circle_color=[159, 72, 88]):
    """preforms the algorithm outlined in readme

    will return None if any step fails
    will return display image, translation, and rotation if successful
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, c = frame.shape

    # determine center point of red circle.
    color_mask = utility.detect_color(frame, circle_color)
    coords = np.argwhere(color_mask)
    if len(coords) == 0:
        return None
    circle_center = np.mean(coords, axis=0).astype(int)[::-1]
    cv2.circle(frame, circle_center, 7, (0, 255, 0), 4)

    # canny edge detection and finding most likely 12 sided symbol
    canny = canny_edge_detection(frame)
    best_approx = find_shape(canny)

    if best_approx is None:
        return None

    points_image = []
    for i in range(0, len(best_approx)):
        points_image.append((best_approx[i][0][0], best_approx[i][0][1]))

    # sort contour points clockwise with first point closest to circle
    points_image = utility.sort_contour(
        circle_center,
        points_image,
    )

    # homography from 2d points to image points
    homography, _ = cv2.findHomography(
        np.array(points_2d).astype(float),
        np.array(points_image).astype(float),
    )

    # homography from image to centered 2d points
    # show how a 2d plane can be stabilized from multiple view points
    display_homography, _ = cv2.findHomography(
        np.array(points_image).astype(float),
        np.array(points_2d).astype(float) + (w // 2, h // 2),
    )
    warped = cv2.warpPerspective(frame, display_homography, (w, h))
    cv2.imshow("warped", warped)

    # extract rotation and translation from homography
    (R, T) = utility.decompose_homography(A, homography)

    if R is None or T is None:
        return None

    T = np.squeeze(np.array(T))
    Rot = utility.decompose_rotation(R)

    cv2.drawContours(frame, [best_approx], 0, (255, 0, 0), 3)
    for i in range(0, len(best_approx)):
        cv2.circle(frame, points_image[i], 3, (i * 30, 0, 255 - i * 20), 3)
        cv2.putText(frame, str(i), points_image[i], font, 0.5, (255, 255, 255))

    text = f"rX: {np.rad2deg(Rot[0]):0.2f} rY: {np.rad2deg(Rot[1]):0.2f} rZ: {np.rad2deg(Rot[2]):0.2f}"
    cv2.putText(frame, text, (20, 20), font, 0.5, (255, 255, 255))
    text = f"tX: {T[0]:0.2f} tY: {T[1]:0.2f} tZ: {T[2]:0.2f}"
    cv2.putText(frame, text, (20, 40), font, 0.5, (255, 255, 255))

    merged = np.hstack((frame, cv2.cvtColor(canny, cv2.COLOR_BAYER_GB2BGR)))

    return merged, R, T


def main(video):
    # 'H' symbol points: scale to desired output units
    symbol_points = utility.symbol_h_points(133.4, 57, 63.5, 16, 19, 70)

    # get first frame dimensions
    frame_geneartor = utility.FrameGenerator(video)
    first_frame = next(frame_geneartor)
    height, width, channels = first_frame.shape

    # scale image
    max_dimension = 700
    scale, _ = utility.resize_image(first_frame, max_dimension)
    s_width = width * scale
    s_height = height * scale

    # create camera matrix
    fov = 72
    fy = abs((s_height / 2) / np.tan(fov / 2))
    fx = fy
    K = utility.camera_matrix(fx, fy, s_width, s_height)
    print(f"camera matrix:\n{K}")

    translations = []
    rotations = []
    for frame in frame_geneartor:
        _, frame = utility.resize_image(frame, max_dimension)

        result = track_frame(K, frame, symbol_points)

        if result is None:
            continue

        image, R, T = result

        if R is not None:
            translations.append(T)
            rotations.append(R)

        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    utility.plot_translation(translations)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, required=True)
    args = parser.parse_args()
    main(args.f)
