import requests as r
import cv2
import numpy as np
import os
from matplotlib import pyplot as pltV
import time
import mediapipe as mp
from processing_utils import draw_styled_landmarsks , mediapipe_detection , extract_keypoints 
from tensorflow import keras
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import json
from cvzone.FaceMeshModule import FaceMeshDetector
import pyttsx3
import cvzone
from keras.utils import img_to_array
from keras.preprocessing import image
from collections import Counter


def action_recognation():
    """
    Taking each 50 sequences of keypoints and predict which action is preformed
    """
    sequence = []
    predictions = []
    threshold = 0.85
    d = 0
    timer = 1 
    order = ['get ready' , 'open your bag' , 'put your book in your bag']
    
    with mp_holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while order:

            # Read feed
            _ , frame = cap.read()
            frame , faces = detector.findFaceMesh(frame , draw= False)

            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]
                # Those are all for drawing
                #cv2.line(frame , pointLeft , pointRight , (0,200,0),3)
                #cv2.circle(frame , pointLeft , 5 , (161,51,62) , cv2.FILLED) 
                #cv2.circle(frame , pointRight , 5 , (161,51,62) , cv2.FILLED) 
                w , _ = detector.findDistance(pointLeft , pointRight) # pixel distance
                W = 6.3 # cm
                # Finding the focal length
                f = 603
                # from by backgound that focal length is chaning depending on the type of the camera
                # for my camera is 603 , so now we are going to measure the depth
                d =  (f * W)  / w + (0.2 * d)
                cvzone.putTextRect(frame ,f'Depth is {int(d)}cm' , (face[10][0] -30 , face[10][1]-30) ,
                                scale=2)
                if d > 250 :
                    engine.say("Please get close !")
                    engine.runAndWait()

                elif d < 100 :
                    engine.say("Please move back !")
                    engine.runAndWait()

                else:
                    if timer >  0:
                        engine.say(f"{order[0]}")
                        engine.runAndWait()
                        time.sleep(0.5)
                        timer -= 1
           
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarsks(mp_drawing , mp_holistic_model ,image, results)
                    
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    
                    sequence.append(keypoints)
                    sequence = sequence[-50:]
                    
                    if len(sequence) == 50:
                        res = action_model.predict(np.expand_dims(sequence, axis=0))[0]
                        predictions.append(np.argmax(res))
                        action_prediction = actions[np.argmax(res)] # open - put - stand
                        
                        
                        if np.unique(predictions[-40:])[0]==np.argmax(res) and res[np.argmax(res)] > threshold and action_prediction in order[0]: 
                            predictions = []
                            sequence = []
                            time.sleep(1)
                            tmp = order.pop(0)
                            engine.say(f"{tmp} is done")
                            print(f"{tmp} is done")
                            engine.runAndWait()
                            timer += 1
                            # URL to send action
                            r.put(f'https://flutter-app-79e87-default-rtdb.firebaseio.com/{action_prediction}.json', json.dumps({'status' : 'Done' }))
                        else :
                            predictions = []
                            sequence = []
                            time.sleep(1)
                            engine.say(f"{order[0]} is not done , try again !")
                            print(f"{order[0]} is not done , try again !")
                            engine.runAndWait()
                            timer += 1
  
                # Show to screen
                cv2.imshow('OpenCV Feed', frame)
                # Break gracefully
                if ( cv2.waitKey(10) & 0xFF == ord('q') )  or len(order) == 0:
                    engine.say(f"You have successfully completed the task ")
                    engine.runAndWait()
                    face_expression()
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            else :
                engine.say("I can't see clearly. Please stand in good lighting !")
                engine.runAndWait()
                sequence = []
                predictions = []
                time.sleep(2)

def face_expression():
    # Read feed
    labels = []
    d = 0
    while cap.isOpened():
        
        _ , frame = cap.read()
        frame , faces = detector.findFaceMesh(frame , draw= False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            w , _ = detector.findDistance(pointLeft , pointRight) # pixel distance
            W = 6.3 # cm
            # Finding the focal length
            f = 603
            # from by backgound that focal length is chaning depending on the type of the camera
            # for my camera is 603 , so now we are going to measure the depth
            d =  (f * W)  / w + (0.2 * d)
            cvzone.putTextRect(frame ,f'Depth is {int(d)}cm' , (face[10][0] -30 , face[10][1]-30) ,
                                scale=2)
            if d > 250 :
                engine.say("Please get close !")
                engine.runAndWait()

            elif d < 100 :
                engine.say("Please move back !")
                engine.runAndWait()

            else:
                # Expression code
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                            
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction_expression = expression_model.predict(roi)[0]
                    label=emotion_labels[prediction_expression.argmax()]
                    labels.append(label)

            if len(labels) == 10 :
                expression = Counter(labels).most_common(1)[0][0]
                r.put(f'https://flutter-app-79e87-default-rtdb.firebaseio.com/status.json', json.dumps({'feel' : expression}))
                r.put(URL ,json.dumps('stop'))
                break
            cv2.imshow('expression window', frame)
        else:
            engine.say("I can't see clearly. Please stand in good lighting !")
            engine.runAndWait()
            time.sleep(2)


    



if __name__ == '__main__':
    actions = np.array(['open' , 'put' , 'ready'])
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    label_map = {label:num for num , label in enumerate(actions)}

    # Loading models
    expression_model = keras.models.load_model("D:\graduation_project\model_expression.h5")
    action_model = keras.models.load_model('action.h5')
    face_classifier = cv2.CascadeClassifier("D:\graduation_project\haarcascade_frontalface_default.xml")

    # URL that control running program
    URL  = 'https://flutter-app-79e87-default-rtdb.firebaseio.com/tasks/task1.json'

    while True:
        if r.get(URL).json() == 'run':
            mp_holistic_model = mp.solutions.holistic
            mp_drawing = mp.solutions.drawing_utils 

            # Setting camera
            cap = cv2.VideoCapture(0)
            # Setting speaker settings
            engine = pyttsx3.init()
            detector = FaceMeshDetector(maxFaces=1)
            # Get a api to determine what action to detect
            action_recognation()
        else:
            print('Program stoped , Run it !')
        time.sleep(3) # number of seconds waiting for action [run or stop]
