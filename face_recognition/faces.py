import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#create a recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

#bring labels from pickle file
labels = {}
with open('labels.pickle', 'rb') as f:
    original_labels = pickle.load(f)
    #invert the lookup
    labels = {v:k for k, v in original_labels.items()}

cap = cv2.VideoCapture(0)

while (True):
    #capture frame by frame
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        #print(x, y, w, h)
        roi_grey = grey[y:y+h, x:x+w] #region of interest for the grey frame
        roi_color = frame[y:y+h, x:x+w]

        img_item = "my-img.png"
        cv2.imwrite(img_item, roi_grey)

        color = (255, 0, 0) #format is BGR - blue, green, red
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h

        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)

        id_, conf = recognizer.predict(roi_grey)

        if conf > 45 and conf <= 85:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name,(x, y), font, 1, color, stroke, cv2.LINE_AA)

    #display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything is done - release the capture
cap.release()
cv2.destroyAllWindows()