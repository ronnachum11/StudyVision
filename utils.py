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
# from fer import FER

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
    width = int(frame.shape[1] * 40 / 100)
    height = int(frame.shape[0] * 40 / 100)
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
    return sum(historic_blinks[-freq:])/time

def calc_focus_values(im, scores, ratios):
    mem_size = 100 # can change this number
    gaze = GazeTracking() # have this in _init_ so you don't recall it so often
    gaze.refresh(im)
    hratio, vratio = gaze.horizontal_ratio(), gaze.vertical_ratio()
    score = 0
    ratio = (None,None)
    if len(scores): score = scores[-1]
    if len(scores) >= mem_size: score-=scores[-1*mem_size]
    if hratio is not None:
        xp = 2*(0.5-hratio)
        yp = 2*(0.75-vratio)
        score+=(xp*xp)+(yp*yp)
        ratio = (xp,yp)
    else: score = None
    return score, ratio

def update_focus_value(score,size_scores): 
    return 10-max(0.0,min(0.5,score/min(100,size_scores)))*20 # change 100 if mem_size changes, size_scores = len(scores) (after appending scores to it)

def update_focus_plots(scores,ratios):
    mem_size = 100 # can change this number
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_title('Areas of Focus') # change title
    temp_data = ratios[-1*mem_size:] + [(0.001,0.002)]
    data = list(zip(*temp_data))
    sns.kdeplot(data[0],data[1], shade=True,color='tab:purple') # change color
    heatmap = mpld3.fig_to_html(fig)
    temp_scores = np.array(scores[-1*mem_size:])
    temp_scores[temp_scores<0.0] = 0.0
    temp_scores[temp_scores>0.5] = 0.5
    if len(scores) >= 2*mem_size:
        for i in range(mem_size):
            temp_scores[i] = 10-(temp_scores[i]/mem_size)*20
    elif len(scores) >= mem_size:
        for i in range(mem_size):
            temp_scores[i] = 10-(temp_scores[i]/min(mem_size,i+1+len(scores)-mem_size))*20
    else:
        for i in range(len(scores)):
            temp_scores[i] = 10-(temp_scores[i]/(i+1))*20
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_title('Focus Over Time') # change title
    plt.plot(temp_scores)
    focus_plot = mpld3.fig_to_html(fig2)
    return focus_plot, heatmap

def update_overall_focus_plots(scores,ratios):
    mem_size = 100
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_title('Areas of Focus') # change title
    temp_data = ratios[:] + [(0.001,0.002)]
    data = list(zip(*temp_data))
    sns.kdeplot(data[0],data[1], shade=True,color='tab:purple') # change color
    heatmap = mpld3.fig_to_html(fig)
    temp_scores = np.array(scores)
    temp_scores[temp_scores<0.0] = 0.0
    temp_scores[temp_scores>0.5] = 0.5
    for i in range(len(scores)):
        temp_scores[i] = 10-(temp_scores[i]/min(mem_size,i+1))*20
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_title('Focus Over Time') # change title
    plt.plot(temp_scores)
    focus_plot = mpld3.fig_to_html(fig2)
    return focus_plot, heatmap

def get_overall_mood(moods):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    overallMood = {emotion:0 for emotion in emotions}
    for mood in moods:
        for emotion in overallMood:
            overallMood[emotion] += mood[emotion]
    if len(moods):
        overallMood = {emotion:round(overallMood[emotion]/len(moods)) for emotion in overallMood}
    return overallMood

def get_mood(img):
    mood = detect_emotions(img)
    return mood

def detect_emotions(img):
    detector = FER()
    result = detector.detect_emotions(img)
    emotion = {emotion:round(result[0]['emotions'][emotion] * 100) for emotion in result[0]['emotions']}
    return emotion

def shouldNotifty(emotions):
    if emotions['angry'] + emotions['disgust'] + emotions['fear'] + emotions['sad'] > 80:
        return True
    return False

def blink_rate_graph(historic_blink_rate, time_per_batch):
    print(len(historic_blink_rate))
    final_time = len(historic_blink_rate) * time_per_batch/60
    time_arr = np.arange(0, final_time, time_per_batch/60)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title('Blink Rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Blink Rate (blinks per second)')
    ax.plot(time_arr, historic_blink_rate, color='purple', label='Your Blink Rate')
    ax.axhline(y=1/6, color='black', linestyle=':', label='Average Blink Rate')
    ax.legend(loc="best")
    graph = mpld3.fig_to_html(fig)
    plt.savefig('foo.png', bbox_inches='tight')
    return graph

def plot_moods(mood):
    labels = [emotion.capitalize() for emotion in mood]
    sizes = [mood[emotion] for emotion in mood]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels = labels,  autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    piechart = mpld3.fig_to_html(fig1)
    return piechart

def plot_overall_mood(moods):
    fig = plt.figure()

    x = np.array([i for i in range(len(moods))])
    
    mood_dict = {emotion:np.array([moods[i][emotion] for i in range(len(moods))]) for emotion in moods[0]} # make sure len is at least 1
    
    for emotion in mood_dict:
        plt.plot(x, mood_dict[emotion], label=emotion.capitalize())

    plt.legend(loc="upper right")
    plt.xlabel('Time')
    plt.ylabel('Strength of Emotion')
    plt.title('Emotion Breakdown Throughout Study Session')

    linechart = mpld3.fig_to_html(fig)
    return linechart