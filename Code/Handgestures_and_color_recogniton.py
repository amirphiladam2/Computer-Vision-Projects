//AUTHOR:AMIR PHILIP ADAM
//ADDED ON:31.03.2024 10:00PM*
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import pyfirmata

# Initialize Arduino board
comport = 'COM8'
board = pyfirmata.Arduino(comport)

# Initialize LED pins
led_pins = ['d:8:o', 'd:9:o', 'd:10:o', 'd:11:o', 'd:12:o']
leds = [board.get_pin(pin) for pin in led_pins]

# Initialize buzzer pin
buzzer_pin = board.get_pin('d:7:o')

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Define hand gesture configurations
gesture_numbers = [0, 1, 2, 3, 4, 5]
gesture_arrays = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

# Define color ranges
color_ranges = {
    "red": ((170, 100, 100), (180, 255, 255)),  # Red color range in HSV format
    "green": ((36, 100, 100), (86, 255, 255)),  # Green color range in HSV format
    "blue": ((100, 100, 100), (130, 255, 255)),  # Blue color range in HSV format
    # Add more color ranges as needed
}

# Define minimum contour area
min_contour_area = 1000  # Adjust as needed

# Function to control LEDs and buzzer based on finger gesture and color detection
def led_and_buzzer(finger_up, is_color_detected):
    for i, led in enumerate(leds):
        if finger_up[i] == 1:
            led.write(1)
        else:
            led.write(0)
    buzzer_pin.write(1) if is_color_detected else buzzer_pin.write(0)
# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

try:
    while True:
        # Read video frame
        success, image = cap.read()

        # Find hands in the frame
        hands, _ = detector.findHands(image)

        # Default values
        fingers_up = [0, 0, 0, 0, 0]

        # Hand gesture recognition
        if hands:
            hand = hands[0]  # Consider only the first hand
            fingers_up = detector.fingersUp(hand)
            num_fingers_up = sum(fingers_up)
            cv2.putText(image, f"{num_fingers_up} Finger(s) Up", (70, 110), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

            for i, gesture in enumerate(gesture_arrays):
                if fingers_up == gesture:
                    cv2.putText(image, str(gesture_numbers[i]), (70, 110), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

        # Color recognition
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        is_color_detected = False
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest_contour = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > min_contour_area:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    is_color_detected = True
                    break  # Exit loop if any color is detected

        # Control LEDs and buzzer
        led_and_buzzer(fingers_up, is_color_detected)

        # Show the image
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Keyboard interrupt")
