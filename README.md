# Capstone
Capstone project: A machine learning model that detects and translates American Sign Language to written English text

Project Overview: 

Problem Area: Mainstream technology used for online group meetings, such as Zoom, Google Meets, etc. do not provide the infrastructure to support Mute people. As a result, they are forced to resort to chat boxes to communicate with others without being able to utilize their ability to speak sign language (specifically American Sign Language).

Proposed Solution: To create a machine learning model that is able to detect and translate American Sign Language (through live feed video) and translate it into written English text.

Impact: This solution would be able to impact millions of lives (estimated around 70 million worldwide according to WHO) as it would enable people who are unable to speak to communicate to others where before they wouldn't have been able to thus completely getting rid of old communication methods amongst mutes and non-mutes such as chat boxes or written messages.

Data set: The data set I will be using for this project will be a custom data set I will create on my own through the utilization of OpenCV and other Python libraries. The data set will have standardized parameters (such as image size) to ensure clean and accurate data is being recorded for each individual hand symbol (representing a specific letter or word). The dataset will be comprised of hundreds of images per symbol to train the model (the exact number of images used to train the model will be determined on calculated success rates later on) and all images (jpg format) will be captured from live footage of myself performing each hand symbol. The data set will be comprised of subfolders that represent a specific letter or word, and each subfolder will be populated with a specified number of jpg images that are the relevant ASL hand symbols to represent that letter/word.
