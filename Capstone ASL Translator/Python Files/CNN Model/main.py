# Ankit Dheendsa, Brainstation - September 2023

# The following code will be used to create a convolutional neural network for American Sign Language image data 
# The purpose of this script is to create, test and save the CNN model so we can see its accuracy as well as save the classifier model to be used in the demo

# Required imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os

# Get the absolute path to the "CNN" folder on your desktop (sometimes the folder wasnt being accessed via relative path so this gets rid of that uncertainty)
desktop_path = os.path.expanduser("~/Desktop")
dataset_dir = os.path.join(desktop_path, "CNN/Data")

# Define image dimensions and batch size (we keep the sizing the same as when the data was created to normalize the images as much as possible)
img_width, img_height = 224, 224

# Here we define our batch size (this is an iterative process that needs to be tested with other hyper parameters for optimal model performance)
batch_size = 32 # We found that a batch size of 32 was suitable for our models purpose

# Next we will create an ImageDataGenerator for data augmentation and preprocessing
# The purpose of this step is to apply various transformations to the image so that the model is more robust and better responds to unseen data that can be encountered later during actual usage
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.1,  # Randomly shift the width of the images
    height_shift_range=0.1,  # Randomly shift the height of the images
    shear_range=0.2,  # Randomly apply shearing transformations
    zoom_range=0.2,  # Randomly zoom in on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in new pixels using the nearest available pixel
)

# Create data generators for training and validation
# This will be useful for creating batches of training and validation data and will use the "batch_size" variable we created earlier

train_generator = datagen.flow_from_directory(
    dataset_dir,  # Path to the data directory (CNN/Data)
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'  # Specify that this is the training subset
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,  # Path to the data directory (CNN/Data)
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='validation'  # Specify that this is the validation subset
)

# Indicing the class labels/index
class_indices = train_generator.class_indices
# Here we print the class indices to see the order (will be important to ensure correct mapping in conjunction with the "labels.txt" file)
print(class_indices)


# Here we define the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3))) #setting it to have 32 filters with each having a size of (3,3) and the input shape is the images width and height
model.add(MaxPooling2D((2, 2))) #After the convolutional layer we add a max pooling layer which reduces the spatial dimensions of the feature maps (makes it more efficient and less prone to overfitting)
model.add(Conv2D(64, (3, 3), activation='relu')) #We repeat the same architecture as seen above for 2 more convolutional and max-pooling layers, each new convolutional layer has 2x the filters of the previous
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# We now flatten layers from the 3-D output gained from the above step (3 convolutional and max-pooling layers) into a 1-D vector (connected layers)
model.add(Flatten())

# We can now incorporate the fully connected layers
model.add(Dense(128, activation='relu')) #Here we add a fully connected/dense layer with 128 "neurons" or units that are used for learning complex patterns with the newly flattened feature data
model.add(Dropout(0.5))  # Dropout for regularization to help with minimizing overfitting risk (reduces reliance on specific neurons to improve generalization)
model.add(Dense(len(train_generator.class_indices), activation='softmax'))  # Number of classes based on folder names are added as neurons using softmax as an activation type (suitable method for mulit-class classification)

# Next we can compile the model
model.compile(
    loss='categorical_crossentropy',  # We use categorical cross-entropy for multi-class classification
    optimizer=Adam(lr=0.001),  # We add this line to adjust the learning rate as needed
    metrics=['accuracy']
)

# Now we are able to train the model using the train_generator we instantiated earlier
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,  # We adjust the number of epochs (same as batch size) iteratively to determine the best value for model training
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

