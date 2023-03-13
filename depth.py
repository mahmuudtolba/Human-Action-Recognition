import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(0)
d = 0
while True:
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





    cv2.imshow('image' , frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()