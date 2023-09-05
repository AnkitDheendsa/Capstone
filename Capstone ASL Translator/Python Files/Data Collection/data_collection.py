import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv

# Create the capture variable using cv2
capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 30
imgSize = 300
#* CHANGE THE FOLDER NAME FOR EACH SIGN (this folder holds image data for various hand signs)
folder = "Data/C"
counter = 0
landmark_counter = 1

# Create a CSV file for storing landmark data
#* CHANGE THE CSV FILE NAME FOR EACH SIGN  (this creates a new csv file if one isnt made, if it is, the data will be appended - hence the mode = "a")
csv_file = open("CSV/C.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image"] + [f"Landmark {i}_X" for i in range(21)] + [f"Landmark {i}_Y" for i in range(21)])

# Same code as original data capturing except with the inclusion of landmarks
while True:
    success, img = capture.read()
    if not success:
        print("Error: Could not read a frame from the webcam.")
        break
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        # instantiates a list of landmarks for the detected hand
        landmarks = hand['lmList']  

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        
        # Here we initialize the row with the image name
        landmark_row = [f"Image_{landmark_counter}"]  
        
        # Here we iterate through landmark values
        for lm_id, lm in enumerate(landmarks):
            x_lm, y_lm = lm[0], lm[1]
            cv2.circle(imgCrop, (x_lm - x + offset, y_lm - y + offset), 8, (0, 255, 0), cv2.FILLED)  # Draw circles at each landmark
            landmark_row.extend([x_lm, y_lm])  # Add X and Y coordinates to the row

        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))  # Resize cropped image to match white canvas size
        imgWhite[:imgSize, :imgSize] = imgCrop

        # Display the image with overlay
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
        print(f"Image {counter} captured with landmarks")
        
        # Save the landmark data to the CSV file
        landmark_row[0] = f"Image_{landmark_counter}"  # Change the image name in the landmark_row
        csv_writer.writerow(landmark_row)
        landmark_counter += 1  # Increment landmark_counter for the next image

    if key == 27:  # Press Esc key to exit
        break

# Close the CSV file
csv_file.close()

capture.release()
cv2.destroyAllWindows()
