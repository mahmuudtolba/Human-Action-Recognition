import numpy as np
import cv2
import os


def explore_model(mp_holistic_model , mp_drawing ):
    """
    Visulaizing the model and how the landmark is placed and return the shape of data
    """
    cap = cv2.VideoCapture(0)
    #so here we are setting mpmodel
    with mp_holistic_model.Holistic(min_detection_confidence=0.5 , min_tracking_confidence=0.5) as holistic :
        while cap.isOpened():
            ret , frame = cap.read()
            
            # making detection
            frame , results = mediapipe_detection(frame , model=holistic)
            # drawing landmarks
            draw_styled_landmarsks(mp_drawing , mp_holistic_model , frame , results)
            #showing to the screen
            cv2.putText(frame, 'Press q to quit', (15,20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Opencv read' , frame)

            if cv2.waitKey(10) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    print(f'Data shape is {extract_keypoints(results).shape}')

def extract_keypoints(results):
    """
    We are extracting our features out of each frame 
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def draw_styled_landmarsks( mp_drawing , mp_holistic_model , frame , results):
    """
    Drawing landmark we extract form holistic model on frame
    """
    # Draw face connections
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic_model.FACEMESH_TESSELATION, 
                            #joint color , thickness , circle radius
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                            #lines color , thinkness
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
                            ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic_model.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
                            )  
    # Draw left hand connections
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic_model.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
                            )   
    # Draw right hand connections  
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic_model.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
                            )  

def mediapipe_detection(frame , model):
    """
    Preprocessing image and making prediction of each frame
    """
    frame  = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) # model expect RGB 
    frame.flags.writeable = False # save some memory while processing
    results = model.process(frame) # making prediction
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR) # converting back to BGR
    return frame , results

def make_dir(DATA_PATH , actions, no_sequences ):
    """ Make directory where we save the data """
    if  os.path.exists(DATA_PATH):
        print('Data already exist take care of overwriting')
    
    else :
        for action in actions:
            for sequence in range(no_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH , action , str(sequence)))
                except Exception as e:
                    print(f'we have {e}')
            print(f'{os.path.join(DATA_PATH , action )} is created')


def collecting_data(mp_holistic_model , mp_drawing, actions , no_sequences ,sequences_length,DATA_PATH ):
    if input('Are you ready for collecting data ? Please press y or n.\n') == 'y':
        if input('''CAUTION!! \n,
         You would overwrite the data you have Please press y or n.\n ''') == 'y':
            cap = cv2.VideoCapture(0)
            #so here we are setting mpmodel
            with mp_holistic_model.Holistic(min_detection_confidence=0.5 , min_tracking_confidence=0.5) as holistic :
                #looping through actions
                for action in actions:
                    #vidos for each action
                    for sequence in range(no_sequences):
                        #frames for each video of each action
                        for frame_num in range(sequences_length):
                            ret  , frame = cap.read()
                            # making detection
                            frame , results = mediapipe_detection(frame , model=holistic)
                            # drawing landmarks
                            draw_styled_landmarsks(mp_drawing , mp_holistic_model , frame , results)

                            if frame_num == 0:
                                cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                                cv2.putText(frame, '{} Video Number {}'.format(action, sequence), (30,30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.imshow('OpenCV Feed' , frame)
                                #that break is for each video of each actions 
                                cv2.waitKey(5000)

                            else :
                                cv2.putText(frame, '{} Video Number {}'.format(action, sequence), (30,30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                # Show to screen
                                cv2.imshow('OpenCV Feed', frame)

                            keypoints = extract_keypoints(results)
                            npy_path = os.path.join(DATA_PATH , action , str(sequence) , str(frame_num))
                            np.save(npy_path , keypoints)

                            if cv2.waitKey(10) & 0xff == ord('q'):
                                break
                cap.release()
                cv2.destroyAllWindows()
        