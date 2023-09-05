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

    

# Sprint 2: 

Data Download: 2000 images and their numerical representations (via landmark positions) were captured for each of the 5 American Sign Language symbols that pertain to the letters "A","N","K","I","T" respectively for a total of 10,000 images/landmark coordinates. This was done by updating the data collection script previously had to now incorporate landmarks on the hands joints (updated version of data collection file has been uploaded as of this sprint).

Stats and Modelling: The statistical analysis and modelling that was conducted is as follows - 

- Basic Descriptive Statistics: Gaining trend insights on landmark coordinate values
  - Mean, median and standard deviation for landmark positions across all 5 of the different signs
- Correlation Analysis: Calculating correlation amongst features (landmarks)
- Logistic Regression: Predicting landmark values via other features
- Confusion Matrix: Evaluating logistic regression performance


# Sprint 3:
Advanced deep learning modelling. Started with a baseline logistic regression model and created a demo script for that then moved on to creating a Convolutional Neural Network for image recognition with image data of different hand signs for the letters "A", "B", "C", "N", "K", "I", "T" and "Hello"
