#!/usr/bin/env python
import cv2
import numpy as np
import math
# Gets passed an approx contour list and finds if all the angles are 90 degrees


def rightA(approx, thresh):
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
        angle = np.arccos(abs(dot) / (np.linalg.norm(l1) *
                                      np.linalg.norm(l2))) * 180 / np.pi
        # angle1 = np.tanh(y2 - y1, x2 - x1)
        # angle2 = np.tanh(y3 - y2, x3 - x2)

        dif = abs(angle - 90)
        error += dif

        if dif > thresh:
            right = False

    return (right, error / AL)


def drawCircles(image, points, radius, color, thick):
    for i in range(0, len(points)):
        cv2.circle(image, tuple(map(tuple, points))[i], radius, color, thick)

    return 1


def detectColor(frame, lower, upper):
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(frame, lower, upper)
    green_mask = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('hello', green_mask)

    redP = np.where(mask == 255)

    red_point = None
    if np.sum(redP) > 0:
        y = np.mean(redP[0])
        x = np.mean(redP[1])
        red_point = (int(x), int(y))
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), 2)

    return (red_point, mask)


def sortContour(red_point, pts_dst):

    num_dst = np.array(pts_dst)
    fir = None
    sec = None
    fir_dis = float("inf")
    sec_dis = float("inf")

    for i in range(0, len(num_dst)):
        cur_dis = np.linalg.norm(np.array(red_point) - num_dst[i])
        if cur_dis < fir_dis:
            sec_dis = fir_dis
            sec = fir
            fir_dis = cur_dis
            fir = i
        elif cur_dis < sec_dis:
            sec_dis = cur_dis
            sec = i

    if fir == 11 and sec == 0:
        pts_dst = pts_dst[fir:] + pts_dst[:fir]
    elif fir == 0 and sec == 11:
        fir = 11 - fir
        pts_dst = pts_dst[::-1]
        pts_dst = pts_dst[fir:] + pts_dst[:fir]
    elif fir < sec:
        pts_dst = pts_dst[fir:] + pts_dst[:fir]
    elif fir > sec:
        fir = 11 - fir
        pts_dst = pts_dst[::-1]
        pts_dst = pts_dst[fir:] + pts_dst[:fir]

    return pts_dst

# This decomposes the homogrpahy matrix
# Takes two inputs
# A - intrinsic camera matrix
# H - homography between two 2D points


def decHomography(A, H):
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


def print2Mat(arrayNum):
    iterNum = iter(arrayNum)
    num = next(iterNum)
    matText = '[' + ' '.join(map(str, num))
    for num in arrayNum:
        matText += ';' + ' '.join(map(str, num))
    matText += ']'
    return matText


def decRotation(R):
    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2]))
    z = math.atan2(R[1, 0], R[0, 0])
    return (x, y, z)
