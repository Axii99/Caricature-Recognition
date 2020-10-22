import numpy as np
from PIL import Image,ImageDraw
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import cv2
import dlib
import glob
import os
import math

def align_face(image_array, landmarks, idx1 = 0, idx2 = 2):
    # top = landmarks[0]
    # bottom = landmarks[2]
    top = landmarks[idx1]
    bottom = landmarks[idx2]
    # calculate the mean point of landmarks of left and right eye

    # compute the angle between the eye centroids
    dy = bottom[1] - top[1]
    dx = bottom[0] - top[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    if idx1 == 0 and idx2 == 2:
        angle = math.atan2(dy, dx) * 180. / math.pi -90
    else:
        angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((top[0] + bottom[0]) // 2,
                  (top[1] + bottom[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

def rotate(origin, point, angle, row):
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)

def rotate_landmarks(landmarks, center, angle, row):

    rotated_landmarks = landmarks
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        rotated_landmark = rotate(origin=center, point=pos, angle=angle, row=row)
        rotated_landmarks[idx] = rotated_landmark
    return rotated_landmarks


def align_img(file):
    path1 = 'D:\WorkSpace\WebCaricature\FacialPoints\\'
    path2 = 'D:\WorkSpace\WebCaricature\OriginalImages\\'
    txt= path1 + file + ".txt"
    jpg = path2 + file +'.jpg'
    print(txt)
    print(jpg)
    with open(txt, 'r') as f:
        landmarks = [[float(num) for num in line.split()] for line in f]


    img = cv2.imread(jpg,1)
    # font = cv2.FONT_HERSHEY_SIMPLEX


    rimg, center, angle=align_face(img,landmarks,0,2)

    # top = landmarks[0]
    # bottom = landmarks[2]
    # cv2.circle(rimg, (top[0],top[1]), 5, color=(0, 255, 0))
    # cv2.circle(rimg, (bottom[0],bottom[1]), 5, color=(0, 255, 0))
    rl = rotate_landmarks(landmarks,center, angle, img.shape[0])
    for idx, point in enumerate(rl):
        pos = (point[0], point[1])
        cv2.circle(rimg, pos, 5, color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rimg, str(idx+1), pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
        print(idx,pos)
    cv2.imshow("rl",rimg)
    cv2.waitKey(0)
    xmax = np.max(np.array(rl)[:,0])
    ymax = np.max(np.array(rl)[:,1])
    maxpos = (xmax,ymax)
    minpos = (np.min(np.array(landmarks)[:,0]),np.min(np.array(landmarks)[:,1]))
# cv2.rectangle(rimg,minpos,maxpos,(0,255,0),5)
    cropimage = rimg[int(minpos[1]*0.8):int(maxpos[1]*1.2), int(minpos[0]*0.8):int(maxpos[0]*1.2)]
    if cropimage.any():
        return cropimage
        #cv2.imshow('a', cropimage)
    else:
        return rimg
    #     cropimage = rimg
    #     cv2.imshow('a', cropimage)
    # cv2.waitKey(0)


def align_img_eyes(file):
    path1 = 'D:\WorkSpace\WebCaricature\FacialPoints\\'
    path2 = 'D:\WorkSpace\WebCaricature\OriginalImages\\'
    txt= path1 + file + ".txt"
    jpg = path2 + file +'.jpg'
    print(txt)
    print(jpg)
    with open(txt, 'r') as f:
        landmarks = [[float(num) for num in line.split()] for line in f]


    img = cv2.imread(jpg,1)
    # font = cv2.FONT_HERSHEY_SIMPLEX


    rimg, center, angle=align_face(img, landmarks, 8, 11)

    # top = landmarks[0]
    # bottom = landmarks[2]
    # cv2.circle(rimg, (top[0],top[1]), 5, color=(0, 255, 0))
    # cv2.circle(rimg, (bottom[0],bottom[1]), 5, color=(0, 255, 0))
    rl = rotate_landmarks(landmarks,center, angle, img.shape[0])
    for idx, point in enumerate(rl):
        pos = (point[0], point[1])
        cv2.circle(rimg, pos, 5, color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rimg, str(idx+1), pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
        print(idx,pos)
    cv2.imshow("rl",rimg)
    cv2.waitKey(0)
    xmax = np.max(np.array([rl[8], rl[9], rl[10], rl[11]])[:,0])
    ymax = np.max(np.array([rl[8], rl[9], rl[10], rl[11]])[:,1])
    maxpos = (xmax,ymax)
    minpos = (np.min(np.array([rl[8], rl[9], rl[10], rl[11]])[:,0]),np.min(np.array([rl[8], rl[9], rl[10], rl[11]])[:,1]))
# cv2.rectangle(rimg,minpos,maxpos,(0,255,0),5)
    cropimage = rimg[int(minpos[1]*0.9):int(maxpos[1]*1.1), int(minpos[0]*0.9):int(maxpos[0]*1.1)]
    if cropimage.any():
        return cropimage
        # cv2.imshow('a', cropimage)
    else:
        return rimg
    #     cropimage = rimg
    #     cv2.imshow('a', cropimage)
    # cv2.waitKey(0)


def align_img_mouth(file):
    path1 = 'D:\WorkSpace\WebCaricature\FacialPoints\\'
    path2 = 'D:\WorkSpace\WebCaricature\OriginalImages\\'
    txt= path1 + file + ".txt"
    jpg = path2 + file +'.jpg'
    print(txt)
    print(jpg)
    with open(txt, 'r') as f:
        landmarks = [[float(num) for num in line.split()] for line in f]


    img = cv2.imread(jpg,1)
    # font = cv2.FONT_HERSHEY_SIMPLEX


    rimg, center, angle=align_face(img,landmarks,14, 16)

    # top = landmarks[0]
    # bottom = landmarks[2]
    # cv2.circle(rimg, (top[0],top[1]), 5, color=(0, 255, 0))
    # cv2.circle(rimg, (bottom[0],bottom[1]), 5, color=(0, 255, 0))
    rl = rotate_landmarks(landmarks,center, angle, img.shape[0])
    # for idx, point in enumerate(rl):
    #     pos = (point[0], point[1])
    #     cv2.circle(rimg, pos, 5, color=(0, 255, 0))
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(rimg, str(idx+1), pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
    #     print(idx,pos)
    # cv2.imshow("rl",rimg)
    # cv2.waitKey(0)
    xmax = np.max(np.array([rl[13], rl[14], rl[15], rl[16]])[:,0])
    ymax = np.max(np.array([rl[13], rl[14], rl[15], rl[16]])[:,1])
    maxpos = (xmax,ymax)
    minpos = (np.min(np.array([rl[13], rl[14], rl[15], rl[16]])[:,0]),np.min(np.array([rl[13], rl[14], rl[15], rl[16]])[:,1]))
# cv2.rectangle(rimg,minpos,maxpos,(0,255,0),5)
    cropimage = rimg[int(minpos[1]):int(maxpos[1]), int(minpos[0]):int(maxpos[0])]
    if cropimage.any():
        return cropimage
        # cv2.imshow('a', cropimage)
    else:
        return rimg
    #     cropimage = rimg
    #     cv2.imshow('a', cropimage)
    # cv2.waitKey(0)

img = align_img_eyes('Bruce Lee\C00001')
cv2.imshow("a", img);
cv2.waitKey(0)