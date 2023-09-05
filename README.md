# Capstone
Capstone project: A machine learning model that detects and translates American Sign Language to written English text

NOTE: Data collection script is a standard python file and not a jupyter notebook. 

Project Overview: 

Problem Area: Mainstream technology used for online group meetings, such as Zoom, Google Meets, etc. do not provide the infrastructure to support Mute people. As a result, they are forced to resort to chat boxes to communicate with others without being able to utilize their ability to speak sign language (specifically American Sign Language).

Proposed Solution: To create a machine learning model that is able to detect and translate American Sign Language (through live feed video) and translate it into written English text.

Impact: This solution would be able to impact millions of lives (estimated around 70 million worldwide according to WHO) as it would enable people who are unable to speak to communicate to others where before they wouldn't have been able to thus completely getting rid of old communication methods amongst mutes and non-mutes such as chat boxes or written messages.

Data set: The data set I will be using for this project will be a custom data set I will create on my own through the utilization of OpenCV and other Python libraries. The data set will have standardized parameters (such as image size) to ensure clean and accurate data is being recorded for each individual hand symbol (representing a specific letter or word). The dataset will be comprised of hundreds of images per symbol to train the model (the exact number of images used to train the model will be determined on calculated success rates later on) and all images (jpg format) will be captured from live footage of myself performing each hand symbol. The data set will be comprised of subfolders that represent a specific letter or word, and each subfolder will be populated with a specified number of jpg images that are the relevant ASL hand symbols to represent that letter/word.


# Project Organization
```
**README.md**             -> readme file for developers usage and understanding of the project
|
**ASL Translator**
│
├── Python Files
│   │
│   ├── Data Collection
│   │   └── data_collection.py
│   │
│   ├── Logistic Regression
│   │   ├── LogReg_Demo.py
│   │   ├── main.py
│   │   ├── trained_logistic_regression_model.joblib
│   │   ├── label_encoder.joblib
│   │   └── CSV
│   │
│   └── CNN Model
│       ├── asl_classifier.h5
│       ├── labels.txt
│       ├── CNN_Demo.py
│       └── main.py
│
└── Jupyter Notebooks
    │
    ├── CNN Model
    │   ├── CNN.ipynb
    │   ├── labels.txt
    │   ├── asl_classifier.h5
    │   └── validation_labels.txt
    │
    └── Logistic Regression
        ├── LogReg.ipynb
        ├── label_encoder.joblib
        ├── trained_logistic_regression_model.joblib
        └── CSV
```
# File Dictionary
```
Python Files: Contains all python files for the project, if you wish to replicate the project or test it you will only need these files
Jupyter Notebooks: Purely for marking purposes only, all jupyter notebooks are copies of original python files but with extra markdown

Python Files/Data Collection/data_collection.py: Is the script used to create all custom image and tabular data for ASL hand signs
Python Files/Logistic Regression/main.py: Is the script used to create the logistic regression on tabular ASL data
Python Files/Logistic Regression/LogReg_Demo.py: Is the script used to demo the logistic regression in realtime via OpenCV and the webcam
Python Files/Logistic Regression/trained_logistic_regression_model.joblib: Is the joblib file to store the logistic regression model
Python Files/Logistic Regression/label_encoder.joblib: Is the joblib file to save the logistic regression encoder
Python Files/Logistic Regression/CSV: Is the CSV folder containing all CSV/tabular subfolders for ASL data
Python Files/CNN Model/main.py: Is the script used to create the CNN
Python Files/CNN Model/labels.txt: Is the txt file to store class label names and indexes
Python Files/CNN Model/CNN_Demo.py: Is the script to demo the CNN via OpenCV and the webcam
Python Files/CNN Model/asl_classifier.h5: Is the file that stores the CNN model

Jupyter Notebooks/CNN Model/validation_labels.txt: Is the file that stores the validation data class labels
Jupyter Notebooks/CNN Model/LogReg.ipynb: Is the file that stores the logistic regression (same as python file)

For all other files in the Jupyter Notebooks folder, they are replicas of the files with the same names as the python versions
```
# Sprint 2: 

Data Download: 2000 images and their numerical representations (via landmark positions) were captured for each of the 5 American Sign Language symbols that pertain to the letters "A","N","K","I","T" respectively for a total of 10,000 images/landmark coordinates. This was done by updating the data collection script previously had to now incorporate landmarks on the hands joints (updated version of data collection file has been uploaded as of this sprint).

Stats and Modelling: The statistical analysis and modelling that was conducted is as follows - 

- Basic Descriptive Statistics: Gaining trend insights on landmark coordinate values
  - Mean, median and standard deviation for landmark positions across all 5 of the different signs
- Correlation Analysis: Calculating correlation amongst features (landmarks)
- Logistic Regression: Predicting landmark values via other features
- Confusion Matrix: Evaluating logistic regression performance


# Sprint 3:
Advanced deep learning modelling. Started with a baseline logistic regression model and created a demo script for that then moved on to creating a Convolutional Neural Network for image recognition with image data of different hand signs for the letters "A", "B", "C", "N", "K", "I", "T" and "Hello". The modelling was conducted as follows: 

- Test logistic regression accuracy on still images
- Test logistic regression with real time webcam video data
- Create CNN and train on still image data
- Test CNN accuracy on still images (via validation data set)
- Test CNN with real time webcam video data
