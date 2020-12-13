import cv2
import dlib
import math
import datetime
import numpy as np
# from gaze_tracking import GazeTracking
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
import time
plt.style.use('dark_background')
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
    # gaze = GazeTracking() # have this in _init_ so you don't recall it so often
    # gaze.refresh(im)
    # hratio, vratio = gaze.horizontal_ratio(), gaze.vertical_ratio()
    hratio, v_ratio = 0.1, 0.1
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
    plt.savefig("heatmap.png", bbox_inches='tight')
    heatmap = mpld3.fig_to_html(fig)

    temp_scores = np.array(scores[-1*mem_size:])
    if len(scores) >= 2*mem_size:
        for i in range(mem_size):
            temp_scores[i] = temp_scores[i]/mem_size
    elif len(scores) >= mem_size:
        for i in range(mem_size):
            temp_scores[i] = temp_scores[i]/min(mem_size,i+1+len(scores)-mem_size)
    else:
        for i in range(len(scores)):
            temp_scores[i] = temp_scores[i]/(i+1)
    temp_scores[temp_scores<0.0] = 0.0
    temp_scores[temp_scores>0.5] = 0.5
    temp_scores = 10-(temp_scores*20)
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_title('Focus Over Time') # change title
    plt.plot(temp_scores, color="purple")
    focus_plot = mpld3.fig_to_html(fig2)

    plt.savefig("focus_line.png", bbox_inches='tight')

    return focus_plot, heatmap

def update_focus_overall_plots(scores,ratios):
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
    for i in range(len(scores)):
        temp_scores[i] = temp_scores[i]/min(mem_size,i+1)
    temp_scores = 10-(temp_scores*100)
    temp_scores[temp_scores<0.0] = 0.0
    temp_scores[temp_scores>10.0] = 10.0
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_title('Focus Over Time') # change title
    plt.plot(temp_scores)
    focus_plot = mpld3.fig_to_html(fig2)
    #plt.show()
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

def shouldNotify(emotions):
    if emotions['angry'] + emotions['disgust'] + emotions['fear'] + emotions['sad'] > 80:
        return True
    return False

def blink_rate_graph(historic_blink_rate, time_per_batch):
    print(len(historic_blink_rate))
    final_time = len(historic_blink_rate) * time_per_batch/60
    time_arr = np.arange(0, final_time, time_per_batch/60)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title('Blink Rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Blink Rate (blinks per second)')
    ax.plot(time_arr, historic_blink_rate, color='purple', label='Your Blink Rate')
    ax.axhline(y=1/6, color='white', linestyle=':', label='Average Blink Rate')
    ax.legend(loc="best")
    graph = mpld3.fig_to_html(fig)

    plt.savefig('blink_rate.png', bbox_inches='tight')
    return graph

def plot_moods(mood):
    labels = [emotion.capitalize() for emotion in mood]
    sizes = [mood[emotion] for emotion in mood]

    fig1, ax1 = plt.subplots(figsize=(14,7))
    ax1.pie(sizes, labels = labels,  autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    plt.savefig("moods_pie.png", bbox_inches='tight')
    piechart = mpld3.fig_to_html(fig1)
    return piechart

def plot_overall_mood(moods):
    fig = plt.figure(figsize=(21,7))

    x = np.array([i for i in range(len(moods))])
    
    mood_dict = {emotion:np.array([moods[i][emotion] for i in range(len(moods))]) for emotion in moods[0]} # make sure len is at least 1
    
    for emotion in mood_dict:
        plt.plot(x, mood_dict[emotion], label=emotion.capitalize())

    plt.legend(loc="upper right")
    plt.xlabel('Time')
    plt.ylabel('Strength of Emotion')
    plt.title('Emotion Breakdown Throughout Study Session')

    plt.savefig("moods_line.png", bbox_inches='tight')
    linechart = mpld3.fig_to_html(fig)
    return linechart


focus = [0.2056213431874808, 0.35289810892836726, 0.44110549328188703, 0.5177736689269753, 0.6409489807183584, 0.7011267584961361, 0.7213660801577281, 0.8244422968703083, 0.8770981348128508, 1.083786821386855, 1.5077182137123055, 1.5361148191453005, 1.6302344197666614, 1.7251372974051922, 1.8773080193230525, 1.9649664160642462, 2.0468055456915333, 2.138835691512815, 2.2328343379324562, 2.2786098329601048, 2.343387497005951, 2.353419195692411, 2.505133813235773, 2.6616086238757934]
ratios = [(-0.14814814814814814, 0.4285714285714286), (-0.13680926916221026, 0.35855263157894735), (-0.2836879432624113, 0.08791208791208782), (-0.1914893617021276, -0.19999999999999996), (-0.27083333333333326, 0.2232142857142856), (-0.18000000000000016, -0.16666666666666652), (-0.14154589371980686, -0.014285714285714235), (-0.303030303030303, 0.10606060606060597), (-0.22946859903381656, 0.0), (0.2160756501182033, -0.3999999999999999), (-0.6294073518379595, -0.16666666666666652), (-0.09900284900284895, -0.13636363636363624), (-0.24049619847939185, 0.19047619047619047), (-0.2841312056737588, 0.11904761904761907), (-0.34042553191489366, -0.19047619047619047), (-0.2764227642276422, 0.10606060606060597), (-0.223609872137972, 0.17843137254901964), (-0.21385017421602792, 0.21517027863777072), (-0.17747858017135876, -0.25), (-0.07599667774086383, -0.19999999999999996), (-0.24048538334252623, 0.08333333333333348), (-0.019638043896804014, 0.09821428571428559), (-0.3756756756756756, 0.102870813397129), (-0.3929487179487179, 0.045454545454545414)]
moods = [{'angry': 1, 'disgust': 0, 'fear': 7, 'happy': 0, 'sad': 16, 'surprise': 1, 'neutral': 76}, {'angry': 12, 'disgust': 0, 'fear': 18, 'happy': 3, 'sad': 38, 'surprise': 7, 'neutral': 23}, {'angry': 5, 'disgust': 0, 'fear': 4, 'happy': 0, 'sad': 46, 'surprise': 1, 'neutral': 44}, {'angry': 21, 'disgust': 0, 'fear': 1, 'happy': 0, 'sad': 28, 'surprise': 0, 'neutral': 49}, {'angry': 2, 'disgust': 0, 'fear': 3, 'happy': 9, 'sad': 9, 'surprise': 0, 'neutral': 77}, {'angry': 0, 'disgust': 0, 'fear': 3, 'happy': 93, 'sad': 3, 'surprise': 0, 'neutral': 0}, {'angry': 10, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 9, 'surprise': 0, 'neutral': 80}, {'angry': 20, 'disgust': 0, 'fear': 4, 'happy': 0, 'sad': 21, 'surprise': 0, 'neutral': 54}, {'angry': 1, 'disgust': 0, 'fear': 1, 'happy': 7, 'sad': 10, 'surprise': 0, 'neutral': 80}, {'angry': 3, 'disgust': 0, 'fear': 2, 'happy': 0, 'sad': 24, 'surprise': 0, 'neutral': 70}, {'angry': 1, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 3, 'surprise': 0, 'neutral': 95}, {'angry': 2, 'disgust': 1, 'fear': 1, 'happy': 6, 'sad': 6, 'surprise': 0, 'neutral': 85}, {'angry': 1, 'disgust': 1, 'fear': 1, 'happy': 3, 'sad': 84, 'surprise': 0, 'neutral': 10}, {'angry': 2, 'disgust': 0, 'fear': 1, 'happy': 0, 'sad': 14, 'surprise': 0, 'neutral': 83}, {'angry': 4, 'disgust': 0, 'fear': 9, 'happy': 4, 'sad': 14, 'surprise': 2, 'neutral': 68}, {'angry': 1, 'disgust': 0, 'fear': 1, 'happy': 0, 'sad': 2, 'surprise': 1, 'neutral': 96}, {'angry': 1, 'disgust': 0, 'fear': 2, 'happy': 0, 'sad': 10, 'surprise': 0, 'neutral': 86}, {'angry': 13, 'disgust': 0, 'fear': 6, 'happy': 0, 'sad': 57, 'surprise': 0, 'neutral': 23}, {'angry': 3, 'disgust': 0, 'fear': 1, 'happy': 0, 'sad': 9, 'surprise': 0, 'neutral': 88}, {'angry': 3, 'disgust': 0, 'fear': 38, 'happy': 1, 'sad': 28, 'surprise': 5, 'neutral': 24}, {'angry': 4, 'disgust': 0, 'fear': 8, 'happy': 0, 'sad': 52, 'surprise': 0, 'neutral': 36}, {'angry': 1, 'disgust': 0, 'fear': 2, 'happy': 48, 'sad': 19, 'surprise': 0, 'neutral': 30}, {'angry': 2, 'disgust': 1, 'fear': 2, 'happy': 33, 'sad': 14, 'surprise': 1, 'neutral': 46}, {'angry': 18, 'disgust': 0, 'fear': 8, 'happy': 32, 'sad': 22, 'surprise': 19, 'neutral': 1}, {'angry': 43, 'disgust': 0, 'fear': 14, 'happy': 0, 'sad': 35, 'surprise': 0, 'neutral': 7}, {'angry': 4, 'disgust': 0, 'fear': 9, 'happy': 1, 'sad': 15, 'surprise': 2, 'neutral': 69}]
blink_rate = [0.10785179862091392, 0.2408032395131186, 0.11903337392121677, 0.3514497108285393, 0.2550634359871541, 0.18784838190590036, 0.3506750274336623, 0.3918545203865449, 0.40364941497898843, 0.39107886786209767, 0.18391770562150783, 0.08501436243299539, 0.17458248201425255, 0.026625697915749764, 0.3481676760793933, 0.03466586611824269, 0.1687698802856253, 0.2629737830689151, 0.14382552275991883, 0.188290653057412]

update_focus_plots(focus, ratios)
plot_overall_mood(moods)
plot_moods(moods[-1])
blink_rate_graph(blink_rate, 25)



