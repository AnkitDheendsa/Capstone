# Ankit Dheendsa, Brainstation - September 2023

# The following code will be used to demo the CNN we created and trained on image data of various ASL hand signs
# This demo script utilizes the saved model created from the script entitled "main.py"

# Required imports
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Next, we will initialize the webcam as a means to give the model live data via different ASL hand signs
capture = cv2.VideoCapture(0)
# Here we initialize the HandDetector from the module we imported above from cvzone
detector = HandDetector(maxHands=1)
# Here we instantiate a classifier variable to store the classifier created from main.py and pass in a labels.txt file as well which stores the labels and indexes of each class the model was trained on
classifier = Classifier("asl_classifier.h5", "labels.txt")

# Variables to be used with image resizing/altering
offset = 20
imgSize = 300

# Initialize variables for prediction smoothing
prediction_history = []  # Store a history of predictions (class indices)
prediction_interval = 10  # Number of frames to keep the same prediction, this way the model is more stable with its predictions even for very similar hand signs

# Here we instantiate a list that stores the class labels (the order is taken from the labels.txt file) as well as their indexes
labels = ["A", "B", "C", "N", "K", "I", "T", "Hello"]

# We keep a counter variable to store the number of frames when a hand is detected so that it can be compared with our prediction_interval variable 
frame_counter = 0

# Creating an infinite while loop to activate the image capture/sizing logic
# Note: The following code was taken from our original "data_collection.ipynb" file as it works the same way with a few edits
while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            # Increase the frame counter by 1
            frame_counter += 1
            prediction, index = classifier.getPrediction(imgWhite)
            
            # Append the current class index to the history
            prediction_history.append(index)

            # If the history is longer than the prediction_interval, remove the oldest prediction
            if len(prediction_history) > prediction_interval:
                prediction_history.pop(0)

            # Calculate the smoothed prediction as the mode (most common index) of the indices in the history
            smoothed_index = max(set(prediction_history), key=prediction_history.count)
            smoothed_prediction = labels[smoothed_index]

            # Display the smoothed prediction
            cv2.putText(img, f"Prediction: {smoothed_prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
        cv2.imshow("Webcam", img)
        key = cv2.waitKey(1)
        
        if key == 27:  # Press "Esc" key to exit
            break

# Releasing the webcam capture device (primarily for freeing up system resources)
capture.release()
# Close all windows created by OpenCV upon running this script (making sure no windows are still running in the background)
cv2.destroyAllWindows()
