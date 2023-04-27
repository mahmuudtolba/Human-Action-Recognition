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

# Set mediapipe model 
def make_prediction(URL):
    """
    Taking each 50 sequences of keypoints and predict which action is preformed
    """
    sequence = []
    predictions = []
    threshold = 0.99
    d = 0
    st = time.time()
    order = ['get ready' , 'open the page' , 'put the book in the bag']

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
                                                
                    # Action code
                    
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
                        
                        
                        if np.unique(predictions[-50:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                                predictions = []
                                sequence = []
                                time.sleep(1)
                                
                                action_prediction = actions[np.argmax(res)] # open - put - stand
                                print(action_prediction) 
                                # URL to send action
                                r.put(f'https://flutter-app-79e87-default-rtdb.firebaseio.com/{action_prediction}.json', json.dumps({'status' : 'Done' }))
                                r.put(f'https://flutter-app-79e87-default-rtdb.firebaseio.com/status.json', json.dumps({'feel' : label}))
                                order.pop(0) 
                                

                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', frame)
                    et = time.time()
                    # Break gracefully
                    if ( cv2.waitKey(10) & 0xFF == ord('q') ) or (et - st > 60) or len(order) == 0:
                        r.put(URL ,json.dumps('stop'))
                        cap.release()
                        cv2.destroyAllWindows()
                        return 
            else :
                engine.say("I can't see clearly. Please stand in good lighting !")
                engine.runAndWait()
                sequence = []
                predictions = []
                time.sleep(2)
        


if __name__ == '__main__':
    actions = np.array(['open' , 'put' , 'ready'])
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    label_map = {label:num for num , label in enumerate(actions)}

    # Loading models
    action_model = keras.models.load_model('action.h5')
    mp_holistic_model = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils 

    # Setting camera
    cap = cv2.VideoCapture(0)
    # Setting speaker settings
    engine = pyttsx3.init()
    detector = FaceMeshDetector(maxFaces=1)
    expression_model = keras.models.load_model("D:\graduation_project\model_expression.h5")
    face_classifier = cv2.CascadeClassifier("D:\graduation_project\haarcascade_frontalface_default.xml")
    
    

    # URL that control running program
    URL  = 'https://flutter-app-79e87-default-rtdb.firebaseio.com/tasks/task1.json'
  
    while True:
        if r.get(URL).json() == 'run':
            make_prediction(URL)
        else:
            print('Program stoped , Run it !')
        time.sleep(3) # number of seconds waiting for action [run or stop]