#AUTHOR:AMIR PHIL ADAM
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize variables
rpoints = [deque(maxlen=1024)]
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
kpoints = [deque(maxlen=1024)]
wpoints = [deque(maxlen=1024)]  # List for eraser points

red_index = 0
blue_index = 0
green_index = 0
black_index = 0
white_index = 0  # Index for eraser points

colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0), (255, 255, 255)]  # Red, Blue, Green, Black, Eraser (White)
brush_size = 5

paintWindow = np.zeros((471, 636, 3)) + 255

cv2.ellipse(paintWindow, (90, 33), (50, 30), 0, 0, 360, (0, 0, 0), 2)
cv2.ellipse(paintWindow, (207, 33), (50, 30), 0, 0, 360, (0, 0, 255), 2)
cv2.ellipse(paintWindow, (322, 33), (50, 30), 0, 0, 360, (255, 0, 0), 2)
cv2.ellipse(paintWindow, (437, 33), (50, 30), 0, 0, 360, (0, 255, 0), 2)
cv2.ellipse(paintWindow, (552, 33), (50, 30), 0, 0, 360, (0, 0, 0), 2)

cv2.putText(paintWindow, "ERASER", (55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (185, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (300, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (420, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLACK", (495, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret = True

# Initialize colorIndex variable
colorIndex = None

while ret:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.ellipse(frame, (90, 33), (50, 30), 0, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(frame, (207, 33), (50, 30), 0, 0, 360, (0, 0, 255), 2)
    cv2.ellipse(frame, (322, 33), (50, 30), 0, 0, 360, (255, 0, 0), 2)
    cv2.ellipse(frame, (437, 33), (50, 30), 0, 0, 360, (0, 255, 0), 2)
    cv2.ellipse(frame, (552, 33), (50, 30), 0, 0, 360, (0, 0, 0), 2)
    
    cv2.putText(frame, "ERASER", (55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (185, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (300, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (420, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (495, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if (thumb[1] - center[1] < 30):
            rpoints.append(deque(maxlen=512))
            red_index += 1
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            kpoints.append(deque(maxlen=512))
            black_index += 1
            wpoints.append(deque(maxlen=512))  # Reset eraser points list
            white_index += 1

        elif center[1] <= 65:
            if (center[0] - 90)**2/50**2 + (center[1] - 33)**2/30**2 <= 1:  # Eraser Button
                colorIndex = 4  # Eraser (White)
            elif (center[0] - 207)**2/50**2 + (center[1] - 33)**2/30**2 <= 1:
                colorIndex = 0  # Red
            elif (center[0] - 322)**2/50**2 + (center[1] - 33)**2/30**2 <= 1:
                colorIndex = 1  # Blue
            elif (center[0] - 437)**2/50**2 + (center[1] - 33)**2/30**2 <= 1:
                colorIndex = 2  # Green
            elif (center[0] - 552)**2/50**2 + (center[1] - 33)**2/30**2 <= 1:
                colorIndex = 3  # Black
        else:
            if colorIndex is not None:
                if colorIndex == 0:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 1:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 2:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 3:
                    kpoints[black_index].appendleft(center)
                elif colorIndex == 4:  # Eraser (White)
                    wpoints[white_index].appendleft(center)

    else:
        rpoints.append(deque(maxlen=512))
        red_index += 1
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        kpoints.append(deque(maxlen=512))
        black_index += 1
        wpoints.append(deque(maxlen=512))  # Reset eraser points list
        white_index += 1

    # Draw on paintWindow and frame
    points = [rpoints, bpoints, gpoints, kpoints, wpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is not None and points[i][j][k] is not None:
                    color = colors[i]
                    if color == (255, 255, 255):  # Eraser
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], color, brush_size*2)  # Thicker brush for eraser
                    else:
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], color, brush_size)
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], color, brush_size)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+'):
        brush_size += 1
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

cap.release()
cv2.destroyAllWindows()
