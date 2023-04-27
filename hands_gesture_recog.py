import cv2
import mediapipe as mp
import pandas as pd  
import os,sys
import numpy as np 
import pyttsx3

def image_processed(hand_img):

    
    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            #print("--->>> ", i)
            if i not in garbage:
                #print("Actiual data ", i)
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[:])
        print(clean)

        for i in range(0, len(clean)):
            #print("--->>>", clean[i].split(':')[1])
            clean[i] = float(clean[i].split(':')[1])
        #print("Clean : ", clean)
        return(clean)
    except Exception as e:
        #exc_type, exc_obj, exc_tb = sys.exc_info()
        #print("Exception.. at line # "+str(exc_tb.tb_lineno)+ " for failed components "+str(e))
        return(np.zeros([1,63], dtype=int)[0])

#-------------------------------------  Main  --------------------------------------#
import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)
#print("SVM : ", svm)


import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    #print("...V... ",data)
    if data[0] > 0:
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1,63))
        print("--->>> V --->>> ",y_pred)
    else:
        y_pred = ['No hand movement']
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    
    # Using cv2.putText() method
    frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
    else:
        engine = pyttsx3.init()
        engine.say(str(y_pred[0]))
        engine.runAndWait()
    

cap.release()
cv.destroyAllWindows()