# Ankit Dheendsa September 2023

# The following code will be used to instantiate and run a logistic regression on the tabular data from different ASL hand signs

# Required imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib

# List of the csv file paths we will be pulling x,y coordinates from
csv_file_paths = ["CSV/A.csv","CSV/N.csv","CSV/K.csv","CSV/I.csv","CSV/T.csv","CSV/HELLO.csv","CSV/B.csv","CSV/C.csv"]

# Here we initialize an empty list to store DataFrames and sign labels
dfs = []
sign_labels = ['A', 'N', 'K', 'I', 'T', 'HELLO', 'B', 'C']

# Next we will load each CSV file, add the "Sign" column, and append to the list as a target column 
# The reason for this is to give the logistic regression a column to refer to as a means of determining if its prediction 
# was accurate or not
for file_path, sign_label in zip(csv_file_paths, sign_labels):
    df = pd.read_csv(file_path)
    df['Sign'] = sign_label
    dfs.append(df)

# Next we will concatenate the DataFrames into a single DataFrame to be passed into the model
combined_df = pd.concat(dfs, ignore_index=True)

# Encode the target labels (since they are string objects we will need to give them numerical representations)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(combined_df['Sign'])

# Extract categorical features ("Image" column has the names of each image, and the "Sign" column has the specific ASL
# sign that the rows data refers to)
X = combined_df.drop(["Image", "Sign"], axis=1)

# Here we split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Convert landmark position columns to numeric values (they are originally stored as string object types)
numeric_columns = X.columns
X_train[numeric_columns] = X_train[numeric_columns].apply(pd.to_numeric, errors='coerce')
X_test[numeric_columns] = X_test[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Print statements to make sure the data being passed in is indeed integer/float values
print(X_train.dtypes)
print(X_test.dtypes)


# As a redundancy measure, if there are any missing values we populate it with the mean for the numerical columns
# The mean as a fill method is suitable for this dataset as we are working with positional data
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Next, we can initialize and train the logistic regression model
model = LogisticRegression(multi_class='auto', solver='lbfgs')

# We can fit the model with a progress bar to see its progress over time (this way we can determine how long it will take
# to train and test approximately, and more so of a convenience factor)
with tqdm(total=100, desc="Training Progress") as pbar:
    for _ in range(100):
        model.fit(X_train, y_train_encoded) 
        pbar.update(1)

# Next we will save the trained model using joblib so that way we can import it into other scripts (we will use this for the demo)
# This way when the script is done running it will create two new files, one for the saved model and another for the saved encoder
model_filename = 'trained_logistic_regression_model.joblib'
joblib.dump(model, model_filename)

# We will also save the label encoder
label_encoder_filename = 'label_encoder.joblib'
joblib.dump(label_encoder, label_encoder_filename)


# Now we can make predictions
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test = label_encoder.inverse_transform(y_test_encoded)

# Finally we can generate classification report to gain insights on the models performance
report = classification_report(y_test, y_pred)
print("Classification Report:\n")
print(report)
