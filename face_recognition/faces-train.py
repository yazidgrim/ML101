import os
from PIL import Image
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

#create a recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#label ids - seems to be a good practice to have labels as IDs rather than names
current_label_id = 0
label_ids = {}

x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_label_id
                current_label_id += 1
            id_ = label_ids[label]

            #Opening an image
            pil_image = Image.open(path).convert("L") #convert to grayscale
            image_array = np.array(pil_image, "uint8") #store as numpy array
            #print(image_array)
            
            #detect regions of interest (faces)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            #loop through faces
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w] #region og interest
                x_train.append(roi) #append to training set
                y_labels.append(id_)

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)


#train the model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")