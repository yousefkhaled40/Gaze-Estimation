import cv2
import numpy as np
import dlib
from math import hypot

font = cv2.FONT_HERSHEY_PLAIN


def mid_point(p1, p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    """Detecting if the eyes are closed or not

    Args:
        eye_points : landmark points
        facial_landmarks : landmarks

    """
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 2)
    hor_line_length = hypot(
        (left_point[0]-right_point[0]) + (left_point[1]-right_point[1]))

    center_top = mid_point(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = mid_point(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))
    # vert_line = cv2.line(frame, center_top, center_bottom, (0,255,0), 2)
    vert_line_length = hypot(
        (center_top[0]-center_bottom[0]) + (center_top[1]-center_bottom[1]))

    ratio = hor_line_length/vert_line_length

    return ratio


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(
                                    eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(
                                    eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(
                                    eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(
                                    eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0,0,255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0:height, 0:width//2]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, width//2:width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Handling dividing by zero error
    if left_side_white == 0:
        gaze_ratio = 5
    elif right_side_white == 0:
        gaze_ratio = 1
    else:
        gaze_ratio = left_side_white/right_side_white

    return gaze_ratio


def gaze_estimation(gaze_ratio, max_gaze_ratio):

    # max_gaze_ratio = max(gaze_ratio, max_gaze_ratio)
    pixel_of_interest = int((gaze_ratio / max_gaze_ratio) * 499)

    if pixel_of_interest > 499:
        pixel_of_interest = 499

    return pixel_of_interest


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

max_gaze_ratio = 0

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    frame = cv2.flip(frame, 1)
    black_img = np.zeros((500, 500, 3), np.uint8)

    # gray for less computation power
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # array for all faces
    
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        landmarks = predictor(gray, face)
        

        # Detect Blinking
        left_eye_ratio = get_blinking_ratio(
            [36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio(
            [42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2
        

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255, 0, 0))
            

        # Gaze Detection
        gaze_ratio_left_eye = get_gaze_ratio(
            [36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio(
            [42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2
        cv2.putText(frame, str(gaze_ratio), (50, 100), font, 2, (0, 0, 255), 3)

        max_gaze_ratio = max(max_gaze_ratio, gaze_ratio)

        estimation = gaze_estimation(gaze_ratio, max_gaze_ratio)
        cv2.circle(black_img, (estimation, 249), radius=5,
                   color=(0, 0, 255), thickness=-1)
        

        """if gaze_ratio <= 1:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio < 2.5:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (255, 0, 0)"""
            

    cv2.imshow("Frame", frame)
    cv2.imshow("Estimation", black_img)
    # cv2.imshow("New Frame", new_frame)
    key = cv2.waitKey(500)
    if key == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
