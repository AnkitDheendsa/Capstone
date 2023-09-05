# Ankit Dheendsa, Brainstation - September 2023

# The following code will be used to demo the logistic regression model on tabular landmark data (x,y coordinates) of hand signs
# This demo script utilizes the saved model and encoder created in the logistic regression code file

# Required imports
import numpy as np
import cv2
import joblib
from cvzone.HandTrackingModule import HandDetector

# Here we are going to load the trained model and encoder that we created with our logistic regression code 
# (refer to the same folder/directory to access that file)
loaded_model = joblib.load("trained_logistic_regression_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Next we will initialize webcam as a means to give the model live data via different ASL hand signs
capture = cv2.VideoCapture(0)

# Here we initialize the HandDetector from the module we imported above from cvzone
detector = HandDetector(maxHands=1)


# Creating an infinite while loop to activate the image capture/sizing logic
# Note: The following code was taken from our original "data_collection.ipynb" file as it works the same way with a few edits
while True:
    # Capture.read() produces two variables, both of which we will store in success and img variables. 
    success, img = capture.read()
    # If the image capture was not successful we can throw a print statement and exit the loop 
    if not success:
        print("Error: Could not read a frame from the webcam.")
        break
    
    # If the image capture was successful we can begin to find the hand using the detector variable we instantiated above
    hands, img = detector.findHands(img)
    
    # Here we work on accessing the landmark data from the image (landmark data = the joint positions of the hand)
    if hands:
        hand = hands[0]
        landmarks = hand['lmList']
        
        # Check if landmarks is not empty
        if landmarks:  
            # Check data type of elements in landmarks (good for debugging to make sure integer type data is being sent)
            for lm in landmarks:
                data_type = type(lm[0])
                print(f"Landmark position {landmarks.index(lm)} data type: {data_type}")
       
        x_coordinates = [lm[0] for lm in landmarks]
        y_coordinates = [lm[1] for lm in landmarks]
        
        # Here we create the feature data with 42 features (21 joints, thus 21 x coordinates and 21 y coordinates)
        feature_data = np.array(x_coordinates + y_coordinates).reshape(1, -1)
        
        # We can now make predictions by passing the feature data into the model an dstoring it in a variable
        # Then we can pass that variable into the label encoder to get a string representation of the hand sign 
        # (the corresponding letter or word)
        predicted_encoded = loaded_model.predict(feature_data)
        predicted_sign = label_encoder.inverse_transform(predicted_encoded)
        
        # We can use cv2 functionality to display the predicted sign at the top left of the image in a green font
        cv2.putText(img, f"Predicted Sign: {predicted_sign[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Displaying the current frame in a window entitled "Webcam"
    cv2.imshow("Webcam", img)
    # This variable waits for a key input so we can add break functionality to exit the program when we are done
    key = cv2.waitKey(1)
    
    # Using the key variable we can now press "Esc" key to exit
    if key == 27:  
        break
# Releasing the webcam capture device (primarily for freeing up system resources)
capture.release()
# Close all windows created by opencv upon running this script (making sure no windows are still running in the background)
cv2.destroyAllWindows()
