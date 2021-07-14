##최종코드

import dlib
import cv2
import time
import numpy as np
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import pandas as pd
import tkinter as tk # Tkinter
import tkinter.font
from PIL import ImageTk, Image # Pillow
import keyboard
import os
import math
from __future__ import division #python 2 코드를 python 3으로 이식하기
# Calibration_Start
class Calibration(object): # 동공 검출 알고리즘, 이진화 임계값
    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self): # 보정이 완료된 경우 true 반환
       
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side): 
        
        if side == 0: # 왼쪽
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1: # 오른쪽
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        # 홍채에서 차지하는 공간의 비율을 반환
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        # 이항성을 위한 최적의 임계값
        
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
      
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
            
        elif side == 1:
            self.thresholds_right.append(threshold)
            
# Calibration_End

#Gaze Tracking_Start
class GazeTracking(object):
   
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        
        self._face_detector = dlib.get_frontal_face_detector()
        cwd = os.path.abspath(os.path.dirname("C:/crawling")) # 본인 위치 넣기      
        model_path = os.path.abspath(os.path.join(cwd, "C:/crawling/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
      
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
       
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):  # count + 1
     
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            
            return (x, y)

    def pupil_right_coords(self):  # count + 1
        
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            
            return (x, y)

    def horizontal_ratio(self):
   
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
      
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
      
        if self.pupils_located:
            global right_counter
            right_counter+=1
            return self.horizontal_ratio() <= 0.1

    def is_left(self):
      
        if self.pupils_located:
            global left_counter
            left_counter+=1
            return self.horizontal_ratio() >= 0.9

    def is_center(self):
      
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def annotated_frame(self):
       
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
#             cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
#             cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
#             cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
#             cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
#Gaze Tracking_End

# Pupil_Start
class Pupil(object):
  
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):

        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass
        
# Pupil_End

# Eye_Start
class Eye(object):
  
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)

        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
      
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
# Eye_End


def extract():##정보 추출 함수
    data = pd.DataFrame(columns=['NAME','TIME','BLINK_CNT','TRACKING_RS','CONDITION','ABSENT'])
    new_data = {'NAME':username, 'TIME': time.strftime('%Y-%m-%d %X', time.localtime(time.time())), 'BLINK_CNT':BLINK_RESULT, 'TRACKING_RS':tracking_total_result, 'CONDITION':SLEEP_RESULT, 'ABSENT':ABSENT_RESULT}
    data = data.append(new_data, ignore_index=True)
    data.to_excel("StudentManagement.xlsx")
    #     textExample.delete(0,"end")
#     textExample.insert(0, text)
def exitfun():##종료 함수
    extract()
    win.destroy()
    cv2.destroyAllWindows()
    cap.release()
def calc_angle(pt1, pt2):#각도 계산 함수
    d = pt1 - pt2
    return cv2.fastAtan2(float(d[1]), float(d[0]))

def eye_aspect_ratio(eye):#눈 가로세로 비율 계산 함수
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def video_play():
    global distract_TOTAL
    global BLINK_RESULT
    global BLINK_COUNTER
    global PREV_TIME
    global BLINK_TOTAL
    global SLEEP_RESULT
    global SLEEP_TOTAL
    global SLEEP_STR
    global ABSENT_COUNTER
    global ABSENT_STATE
    global ABSENT_RESULT
    global left_counter
    global right_counter
    global total_counter
    global tracking_total_result
    global tracking_str
    global tracking_total_result_tmp
    ret, frame = cap.read() # 프레임이 올바르게 읽히면 ret은 True
#     frame = imutils.resize(frame, width=600, height=700)
    panel = None
    if not ret:
        win.destroy()
        cv2.destroyAllWindows()
        cap.release() # 작업 완료 후 해제
        return
    frame0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detector = detector(frame, 0)
    cv2.putText(
        frame0,
        "BLINK_CNT: {}".format(BLINK_TOTAL),
        (400, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    blink_label.set(int(BLINK_RESULT))
    cv2.putText(
        frame0,
        "TRACKING_RS: {}".format(tracking_total_result_tmp),
        (400, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    tracking_label.set(str(tracking_str))
    cv2.putText(
        frame0,
        "CONDITION: {}".format(SLEEP_TOTAL),
        (400, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    sleep_label.set(str(SLEEP_STR))

    cv2.putText(
        frame0,
        "ABSENT: {}".format(ABSENT_COUNTER),
        (400, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    absence_label.set(str(ABSENT_STATE))
    gaze.refresh(frame0)

    frame0 = gaze.annotated_frame()
    text = ""
    
    if gaze.is_right():
        text = "Right"
    elif gaze.is_left():
        text = "Left"
    elif gaze.is_center():
        text = "Center"
    
#     cv2.putText(frame0, str(text), (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)#텍스트 출력안됨

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    total_counter = left_counter + right_counter
        #부재 여부 체크
    if len(face_detector) == 0:
        ABSENT_STATE = 'absent'
        ABSENT_COUNTER += 1
        cv2.rectangle(frame0, (0,0), (638,479), (255, 0, 0), 3)
    else:
        ABSENT_STATE = 'Attendance'
                      
    for face in face_detector:
        
        # face wrapped with rectangle
        cv2.rectangle(frame0, (face.left()-30, face.top()-30), (face.right()+15, face.bottom()+15),
                      (0, 255, 0), 3)
        landmarks = predictor(frame0, face)  
        landmarks = face_utils.shape_to_np(landmarks)

        #얼굴에서 68개 점 찾아 배열에 저장
        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        nose = landmarks[nStart:nEnd]
        
        #코를 기준으로 얼굴 기울기 계산
        angle = calc_angle(nose[1], nose[3])
        #기울기로 현재 상태 갱신 : sleep, wake
        if not 245 < angle < 290:
            SLEEP_TOTAL += 1

            
        #ear : 눈 가로 세로 비율 계산
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        
        #ear이 기준치 미만이면 count + 
        if ear < EYE_AR_THRESH:
            BLINK_COUNTER += 1
        if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
            BLINK_TOTAL += 1
            BLINK_COUNTER = 0
    
    # 1분 경과
    CUR_TIME = time.time() - PREV_TIME
    if CUR_TIME > 10:
        PREV_TIME == CUR_TIME
        avg_lf = total_counter/2
        if avg_lf > 100:
            tracking_total_result +=1 # 출력하는 데이터 기준
            tracking_total_result_tmp +=1
            left_counter = 0
            right_counter = 0
            total_counter = 0
    if CUR_TIME > 60:
        if 30 < BLINK_TOTAL:
            BLINK_RESULT += 1
        BLINK_TOTAL = 0
        PREV_TIME = time.time()
        if 20<SLEEP_TOTAL:
            SLEEP_RESULT +=1
            SLEEP_STR = "\U0001F62A"
        else:
            SLEEP_STR = "\U0001f600"
        SLEEP_TOTAL = 0

        if ABSENT_COUNTER > 30:
            ABSENT_RESULT += 1
            ABSENT_COUNTER = 0
        PREV_TIME = time.time()
        if tracking_total_result_tmp < 6:
            tracking_str ="\U0001F62A"
        else:
            tracking_str ="\U0001f610"
        tracking_total_result_tmp = 0
#     cv2.putText(frame0, str(CUR_TIME), (90, 420), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2) 
#     cv2.putText(frame0, str(tracking_str),(90, 300), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 2) 
#     cv2.putText(frame0, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1)
#     cv2.putText(frame0, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 1)
#     cv2.putText(frame0, "Total counter : " +  str(total_counter),(90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
#     cv2.putText(frame0, "Tracking Total counter : " +  str(tracking_total_result),(90, 250), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)            
    img = Image.fromarray(frame0) # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img) # ImageTk 객체로 변환
    # OpenCV 동영상
    lbl1.imgtk = imgtk
    lbl1.configure(image=imgtk)
    key = cv2.waitKey(1)
    lbl1.after(80,video_play)#50~80사이

##username 입력
username = input()

#BLINK, SLEEP변수
BLINK_COUNTER = 0
BLINK_TOTAL = 0
BLINK_RESULT = 0
SLEEP_RESULT = 0
SLEEP_TOTAL = 0
SLEEP_STR = "\U0001f600"
ABSENT_COUNTER = 0
ABSENT_RESULT = 0
ABSENT_STATE = ''

#Eye tracking 변수
gaze = GazeTracking()
left_counter = 0
right_counter = 0
total_counter = 0
tracking_total_result = 0
tracking_total_result_tmp =0
tracking_str = '\U0001F610'


#눈 기준 값
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3


#특정 랜드마크 추출 인덱스
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

#현재 시간
PREV_TIME = time.time()


# GUI 설계
win = tk.Tk() # 인스턴스 생성

win.title("Student Management") # 제목 표시줄 추가
win.geometry("880x550") # 지오메트리: 너비x높이
win.resizable(False, False) # x축, y축 크기 조정 비활성화
font=tkinter.font.Font(family="Times", size=15)#폰트 설정
font_img=tkinter.font.Font(family="Times",size=25)#폰트 설정
#제목 라벨 설정
lbl = tk.Label(win, text="학생 집중 관리 플랫폼",font = font)
lbl.grid(row=0, column=0, columnspan = 1)

# 측정 값 라벨 선언
lbl_name = tk.Label(win, text="NAME",font=font)
lbl_name.place(x=650, y=50)
lbl_blink = tk.Label(win, text="BLINK_CNT",font=font)
lbl_blink.place(x=650, y=100)
lbl_tracking = tk.Label(win, text="TRACKING",font=font)
lbl_tracking.place(x=650, y=150)
lbl_sleep = tk.Label(win, text="CONDITION",font=font)
lbl_sleep.place(x=650, y=200)
lbl_absence = tk.Label(win, text="ABSENT",font=font)
lbl_absence.place(x=650, y=250)


# 측정 값 표시 라벨 선언
name_label=tk.StringVar()
blink_label=tk.StringVar()
tracking_label=tk.StringVar()
sleep_label=tk.StringVar()
absence_label=tk.StringVar()

# name 값 표현
names=tk.Label(win,textvariable=name_label,font=font)
names.place(x=780, y=50)
name_label.set(str(username))
# blink_count 값 표현
blink_count=tk.Label(win,textvariable=blink_label,font=font)
blink_count.place(x=780, y=100)
blink_label.set(int(BLINK_RESULT))
# distract_count 값 표현
tracking_count=tk.Label(win,textvariable=tracking_label,font=font_img)
tracking_count.place(x=780, y=140)
tracking_label.set(str(tracking_str))
# sleep_count 값 표현
sleep_count=tk.Label(win,textvariable=sleep_label,font=font_img)
sleep_count.place(x=780, y=190)
sleep_label.set(int(SLEEP_RESULT))
# absence_count 값 표현
absence_count=tk.Label(win,textvariable=absence_label,font=font)
absence_count.place(x=780, y=250)
absence_label.set(str(ABSENT_STATE))



##정보 추출, 종료 버튼 선언
Data_extract = tk.Button( text = "정보추출", command = lambda :extract(), height = 2, width = 15,font=font)
Data_extract.place(x=670, y=320)
exit = tk.Button( text = "종료", command = lambda :exitfun(), height = 2, width = 15,font=font)
exit.place(x=670, y=400)

# 영상 삽입 프레임 선언   
frm = tk.Frame(win, bg="white", width=600, height=500) # 프레임 너비, 높이 설정
frm.grid(row=1, column=0) # 격자 행, 열 배치
lbl1 = tk.Label(frm)

lbl1.grid()


cap = cv2.VideoCapture(0)

# face detector, predictor 선언
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
   

video_play()
win.mainloop() #GUI 시작