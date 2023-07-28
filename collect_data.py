# We start off by pip installing the packages cvzone and mediapipe
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
# Create the capture variable using cv2, note that 0 is the webcam number (this uses the webcam so we can start using it to create our custom data set)
capture = cv2.VideoCapture(0)
# Creating detector variable to detect hands (will be used for data collection of single hand ASL) - note: we set maxHands=1 because we are only looking to track 1 hand
detector = HandDetector(maxHands=1)

offset = 30
imgSize = 300

# Creating a folder variable where we would like to store the images (this will change everytime we would like to store new image sets in a new folder, for example when we are capturing img data to train for the letter B)
folder = "Data/A"
# Creating a variable to store a count, this count will tell us how many images we have saved
counter = 0
# Creating while loop to turn on webcam
while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    
    # Here we are creating a cropped box that capures only the hand rather than the entire video image (which would also include the background etc.) this way the model is only looking at the hand
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        
        # Creating a white box to keep a standard size of imagery so that it doesnt change in accordance to the height and width of the hand position
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset: y+h+offset, x-offset:x+w+offset]
        
        #Overlaying the white box behind the cropped box
        imgCropShape = imgCrop.shape
        
        
        aspectRatio = h/w
        
        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
             k = imgSize/w
             hCal = math.ceil(k*h)
             imgResize = cv2.resize(imgCrop, (imgSize,hCal))
             imgResizeShape = imgResize.shape
             hGap = math.ceil((imgSize-hCal)/2)
             imgWhite[hGap:hCal+hGap,:] = imgResize
        
        # cv2.imshow("imgCrop" ,imgCrop)
        cv2.imshow("imgWhite" ,imgWhite)
        
    cv2.imshow("Image" ,img)
    key = cv2.waitKey(1) #1ms delay
    
    if key == ord("s"):
        counter += 1 
        cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
        print(f"Image {counter} captured")
