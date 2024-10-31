import torch
import torchvision
from ultralytics import YOLO
import cv2
import cvzone
import math

cap=cv2.VideoCapture(0) 
cap.set(3,1080)
cap.set(4,720)        

#cap=cv2.VideoCapture("videos/video1.mp4")
model=YOLO("../../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
              "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
              "hair drier", "toothbrush", "hairbrush", "paper", "guitar", "trumpet", "violin", "drum", "piano", "flute", 
              "trombone", "saxophone", "lawn mower", "rocket", "star", "moon", "planet", "airplane", "submarine", "whale",
              "shark", "dolphin", "octopus", "crab", "lobster", "coral", "seahorse", "starfish", "turtle", "frog", "snake",
              "lizard", "tiger", "lion", "monkey", "kangaroo", "panda", "koala", "beetle", "ant", "butterfly", "spider", 
              "snail", "owl", "eagle", "duck", "bat", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
              "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
              "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
              "raspberry pi", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hairbrush", "paper","guitar",
              "trumpet", "violin", "drum", "piano", "flute", "trombone", "saxophone", "lawn mower", "rocket", "star", "moon",
              "planet", "airplane", "submarine", "whale", "shark", "dolphin", "octopus", "crab", "lobster", "coral", "seahorse",
              "starfish", "turtle", "frog", "snake", "lizard", "tiger", "lion", "monkey", "kangaroo", "panda", "koala", "beetle",
              "ant", "butterfly", "spider", "snail", "owl", "eagle", "duck", "bat", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
              "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
              "air_conditioner","auto","curtain","door","floor","furniture","house","kitchen","lamp","ceiling","picture","plant","stappler",
]

while True:
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            
            #Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            w,h = x2-x1, y2-y1
            x1, y1, w, h= int(x1), int(y1), int(x2), int(y2)
            cvzone.cornerRect(img,(x1,y1,w,h))
            
            #confidence
            conf=math.ceil((box.conf[0]*100))/100
            
            #class Name
            cls=int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1,thickness=2)
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
