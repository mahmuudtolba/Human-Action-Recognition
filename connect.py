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

# Set mediapipe model 
def make_prediction(URL):
    """
    Taking each 50 sequences of keypoints and predict which action is preformed
    """
    sequence = []
    predictions = []
    threshold = 0.99
    st = time.time()

    with mp_holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() :

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
                    engine.say("Please move back")
                    engine.runAndWait()

                else:
                    """
                    PUT YOUR CODE HERE 'image' IS THE FRAME TO INSERT TO THE MODEL
                    """
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarsks(mp_drawing , mp_holistic_model ,image, results)
                    
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-50:]
                    
                    if len(sequence) == 50:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predictions.append(np.argmax(res))
                        
                        
                        if np.unique(predictions[-50:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                                predictions = []
                                sequence = []
                                time.sleep(1)
                                
                                prediction = actions[np.argmax(res)] # open - put - stand
                                print(prediction)
                                r.put(f'https://flutter-app-79e87-default-rtdb.firebaseio.com/{prediction}.json', json.dumps({'status' : 'Done' }))
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    et = time.time()
                    # Break gracefully
                    if ( cv2.waitKey(10) & 0xFF == ord('q') ) or (et - st > 30) :
                        r.put(URL ,json.dumps('stop'))
                        return 
            else :
                engine.say("Can't see a person , waiting two seconds")
                engine.runAndWait()
                sequence = []
                predictions = []
                time.sleep(2) # i don't like you
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    actions = np.array(['open' , 'put' , 'stand' ])
    label_map = {label:num for num , label in enumerate(actions)}

    # Loading models
    model = keras.models.load_model('action.h5')
    mp_holistic_model = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils 

    # Setting camera
    cap = cv2.VideoCapture(0)
    # Setting speaker settings
    engine = pyttsx3.init()
    detector = FaceMeshDetector(maxFaces=1)
    d = 0

    # URL that control running program
    URL  = 'https://flutter-app-79e87-default-rtdb.firebaseio.com/tasks/task1.json'

    while True:
        if r.get(URL).json() == 'run':
            make_prediction(URL)
        else:
            print('Program stoped , Run it !')
        time.sleep(2) # number of seconds waiting