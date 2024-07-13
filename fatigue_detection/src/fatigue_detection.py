#!/usr/bin/env python

import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
import numpy as np
import threading
import time
import pyttsx3
from gtts import gTTS
import socket
import time

def draw_landmarks(image, outputs, land_mark, color): ## This function draws facial landmarks
    height, width =image.shape[:2]

    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]

        point_scale = ((int)(point.x * width), (int)(point.y*height))

        cv.circle(image, point_scale, 1, color, 2)

def euclidean_distance(image, top, bottom): ## This function calculates euclidean distance of the specific points or landmarks
    height, width = image.shape[0:2]

    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)

    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, outputs, top_bottom, left_right): ## This function calculates aspect ratio
    landmark = outputs.multi_face_landmarks[0]

    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]

    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]

    left_right_dis = euclidean_distance(image, left, right)

    aspect_ratio = left_right_dis/ top_bottom_dis

    return aspect_ratio

face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0,0,255), thickness=3, circle_radius=2)
connection_style = draw_utils.DrawingSpec((0,0,255), thickness=3, circle_radius=2)

STATIC_IMAGE = False
MAX_NO_FACES = 1
DETECTION_CONFIDENCE = 0.8#0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)
data_flag=0

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
noseTip= [151]

RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

LEFT_EYE_TOP_BOTTOM = [386, 374]

LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]

RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces= MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Error: Can't access the camera.")
    exit()

frame_counter = 0
frame_count = 0
min_frame = 6
min_tolerance = 5.0
start_point = (640, 0)
end_point = (0, 480)
previous_centers = []
flag_condition =0
fps = 0
frame_count_fps = 0
start_time = time.time()
speech = pyttsx3.init()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 12345))

while True:
    result, image = capture.read()
    image = cv.flip(image, 1)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    elapsed_time = time.time() - start_time
    frame_count_fps += 1

    image = cv.line(image, (18, 10), (270, 10), (0, 255, 0), 2)
    image = cv.line(image, (18, 70), (270, 70), (0, 255, 0), 2)
    image = cv.line(image, (220, 80), (220, 480), (51, 255, 255), 2)
    image = cv.line(image, (470, 80), (470, 480), (51, 255, 255), 2)

    cv.putText(image, 'Condition:', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)


    if elapsed_time > 1:
        fps = frame_count_fps / elapsed_time
        start_time = time.time()
        frame_count_fps = 0

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)


            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            radius = int(min(bbox[2], bbox[3]) // 2)


            cv.circle(image, (center_x, center_y), radius, (255, 0, 255), 2)
            cv.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

            cv.line(image, (0,center_y), (center_x, center_y), (255, 0, 0), 2)
            cv.line(image, (center_x, 0), (center_x, center_y), (255, 0, 0), 2)

            cv.line(image, (640,center_y), (center_x, center_y), (255, 0, 0), 2)
            cv.line(image, (center_x, 480), (center_x, center_y), (255, 0, 0), 2)

            if center_x< 200 or center_x> 470:
                flag_condition = 2
            else:
                flag_condition = 0

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)
        frame_counter += 1

        if outputs.multi_face_landmarks:

            draw_landmarks(image, outputs, FACE,(255,217,25) )
            draw_landmarks(image, outputs, noseTip, COLOR_GREEN)


            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_GREEN)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, (255,248,240))

            ratio_left =  get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)


            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_GREEN)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, (255,248,240))

            draw_landmarks(image, outputs, UPPER_LOWER_LIPS ,COLOR_RED )
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, (255,248,240))

            ratio_right =  get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
            ratio_lips = get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

            ratio = (ratio_left + ratio_right)/2.0
            cv.putText(image, f"FPS: {fps:.2f}", (20, 350), cv.FONT_HERSHEY_SIMPLEX, 0.5, (204, 102, 0), 2,
                           cv.LINE_AA)
            cv.putText(image, f"Coordinate: {int(center_x)};{int(center_y)}", (20, 370), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (204, 102, 0), 2, cv.LINE_AA)

            print(f"{data_flag};{flag_condition}")

            if ratio > min_tolerance:
                frame_count +=1
            else:
                frame_count = 0
                
		#Detection data
            if frame_count > min_frame or ratio_lips < 3:
                data_flag = 2
                data='Fatigue'
                cv.putText(image, data, (180, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            else:
                data_flag = 0
                data='Wake'
                cv.putText(image, 'Wake', (180, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            
            message = f"{data_flag};{flag_condition}"
            client_socket.send(message.encode())
            print(f"Sent: {message}")
            #time.sleep(1) 

        cv.imshow("Safe.in Driver device", image)
        if cv.waitKey(1) & 255 == 27:
            break

capture.release()
cv.destroyAllWindows()

