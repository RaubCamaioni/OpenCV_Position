#!/usr/bin/env python
import cv2
import numpy as np
import shapeUtil as su
import scipy.io as sio
import matplotlib.pyplot as plt
import math


red_point = (0, 0)
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
#out = cv2.VideoWriter('AAA.avi', fourcc, 10.0, (1280, 480))


def trackFrame(A, frame, pts_src, base):

    #global out
    global red_point
    R, T = None, None
    blurr = cv2.GaussianBlur(frame, (5, 5), 0)
    imgG = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
    imgC = cv2.Canny(imgG, 50, 60)
    imgC = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (3, 3))
    # imgC = cv2.dilate(imgC, (3, 3), iterations=2)
    (cont, _) = cv2.findContours(imgC.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_approx = None
    lowest_error = float("inf")

    for c in cont:
        pts_dst = []
        perim = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, .01 * perim, True)
        area = cv2.contourArea(c)

        if len(approx) == 12:
            right, error = su.rightA(approx, 40)
            # print(right)
            if error < lowest_error and right:
                lowest_error = error
                best_approx = approx

    # red_point, _ = su.detectColor(blurr, red_lower, red_upper)
    # if red_point is not None:
       # cv2.circle(frame, (red_point[0] + frame.shape[0] / 2, red_point[1] + frame.shape[1] / 2), 5, (0, 0, 255), 2)

    if best_approx is not None:

        cv2.drawContours(frame, [best_approx], 0, (255, 0, 0), 3)

        for i in range(0, len(best_approx)):
            pts_dst.append((best_approx[i][0][0], best_approx[i][0][1]))
            # cv2.circle(frame, pts_dst[-1], 3, (i*30, 0, 255-i*20), 3)

        # Correction method for contour points.  Need to make sure the points are mapped correctly
        # print pts_dst[0]
        pts_dst = su.sortContour(
            np.array((red_point[0] + pts_dst[0][0], red_point[1] + pts_dst[0][1])), pts_dst)

        cv2.circle(frame, pts_dst[0], 7, (0, 255, 0), 4)
        cv2.circle(frame, pts_dst[1], 7, (200, 0, 200), 4)

        for i in range(0, len(best_approx)):
            cv2.circle(frame, pts_dst[i], 3, (i * 30, 0, 255 - i * 20), 3)

        if cv2.waitKey(1) & 0xFF == ord('n'):
            base = frame
            matString = su.print2Mat(pts_src)

        h, status = cv2.findHomography(np.array(pts_src).astype(float), np.array(pts_dst).astype(float))

        #warped = cv2.warpPerspective(base, h, (base.shape[1], base.shape[0]))
        #cv2.imshow('warped', warped)

        (R, T) = su.decHomography(A, h)
        Rot = su.decRotation(R)
        zR = np.matrix([[math.cos(Rot[2]), -math.sin(Rot[2])], [math.sin(Rot[2]), math.cos(Rot[2])]])
        cv2.putText(imgC, 'rX: {:0.2f} rY: {:0.2f} rZ: {:0.2f}'.format(Rot[0] * 180 / np.pi, Rot[1] * 180 / np.pi, Rot[2] * 180 / np.pi), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
        cv2.putText(imgC, 'tX: {:0.2f} tY: {:0.2f} tZ: {:0.2f}'.format(T[0, 0], T[0, 1], T[0, 2]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
        pDot = np.dot((-200, -200), zR)
        red_point = (int(pDot[0, 0]), int(pDot[0, 1]))
        cv2.circle(frame, (int(pDot[0, 0]) + pts_dst[0][0], int(pDot[0, 1]) + pts_dst[0][1]), 5, (0, 0, 255), 2)

    # cv2.imshow('base', base)
    # print frame.shape
    # print imgC.shape
    # cv2.imshow('frame', frame)
    # cv2.imshow('imgC', imgC)
    # cv2.imshow('imgG', imgG)

    merged = np.concatenate((frame, cv2.cvtColor(imgC, cv2.COLOR_BAYER_GB2BGR)), axis=1)
    #out.write(merged)
    cv2.imshow('merged', merged)

    if R is not None:
        return (R, T)
    else:
        return None


if __name__ == "__main__":

    last_center = (0, 0)
    mat = sio.loadmat('camera.mat')
    A = mat['A']
    A = np.matrix([[476.7, 0.0, 400.0], [0.0, 476.7, 400.0], [0.0, 0.0, 1.0]])
    red_lower = [115, 100, 100]
    red_upper = [125, 255, 255]

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('output.avi')
    
    pts_src = np.array([[0.0, 87.0], [22.0, 87.0], [22.0, 50.0],  # pixel measurements, just scaled ratio of sides  
                        [56.0, 50.0], [56.0, 87.0], [78.0, 87.0],
                        [78.0, 0], [56.0, 0], [56.0, 33.0],
                        [22.0, 33.0], [22.0, 0.0], [0, 0]])

    # pts_src = pts_src / 166.5  # convert pixels to meters, can be changed for different sized "H"
    pts_src = pts_src[::-1]  # reverse the order of the array

    rotations = []  # create a structure to store information for matlab
    trans = []

    while(True):
        _, frame = cap.read()

        posRot = trackFrame(A, frame, pts_src, frame)
        if posRot is not None:
            R, T = posRot
            rotations.append(R)
            trans.append(T)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sio.savemat('rotations.mat', {'rotations': np.array(rotations), 'translation': np.array(trans)})
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
