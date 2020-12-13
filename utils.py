import cv2
import dlib
import math
import datetime
import numpy as np
from gaze_tracking import GazeTracking
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
import time

BLINK_RATIO_THRESHOLD = 7

def midpoint(point1 ,point2):
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)
    ratio = horizontal_length / vertical_length
    return ratio

def get_blink_rate(frame, time, blink_ratio_threshold, historic_blinks):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]
    blinks = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * 20 / 100)
    height = int(frame.shape[0] * 20 / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    faces, scores, _ = detector.run(image = frame, upsample_num_times = 0, adjust_threshold = 0.0)
    face = faces[scores.index(max(scores))]
    landmarks = predictor(frame, face)
    left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
    right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
    blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
    if blink_ratio > blink_ratio_threshold:
        historic_blinks.append(1)
    else:
        historic_blinks.append(0)
    return historic_blinks

def current_blink_rate(historic_blinks, freq, time):
    return str(sum(historic_blinks[-freq:])/time)

def focus_method(im, scores, ratios, heatmap):
    mem_size = 100 # can change this number
    gaze = GazeTracking() # have this in _init_ so you don't recall it so often
    gaze.refresh(im)
    hratio, vratio = gaze.horizontal_ratio(), gaze.vertical_ratio()
    score = scores[-1]
    ratio = (None,None)
    if len(scores) >= mem_size: score-=scores[-1*mem_size]
    if hratio is not None:
        xp = 2*(0.5-hratio)
        yp = 2*(0.75-vratio)
        score+=(xp*xp)+(yp*yp)
        ratio = (xp,yp)
        if len(scores) % mem_size == 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_title('Areas of Focus') # change title
            temp_data = ratios[1-mem_size:] + [ratio]
            data = list(zip(*temp_data))
            sns.kdeplot(data[0],data[1], shade=True,color='tab:purple') # change color
            heatmap = mpld3.fig_to_html(fig)
    else: score = None
    return score, ratio, heatmap

def get_focus_value(scores):
    return 10-max(0.0,min(0.5,scores[-1]/min(100,len(scores))))*20 # change 100 if mem_size changes
