# Tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load the TNBC data
tnbc_file_path = 'newData.xlsx'
tnbc_data = pd.read_excel(tnbc_file_path, skiprows=1)

# Rename the columns based on the first row
tnbc_data.columns = ['no', 'BRCA1', 'BRCA2', 'TP53', 'EGFR (HER1)', 'MET', 'RB1', 'PIK3CA', 'subtype']
tnbc_data = tnbc_data.drop(columns=['no'])

# Load the normal data
normal_file_path = 'normal_data.xlsx'
normal_data = pd.read_excel(normal_file_path, skiprows=1)

# Rename the columns based on the first row
normal_data.columns = ['no', 'BRCA1', 'BRCA2', 'TP53', 'EGFR (HER1)', 'MET', 'RB1', 'PIK3CA']
normal_data = normal_data.drop(columns=['no'])
normal_data['subtype'] = 'normal'  # Add a new label for normal patients

# Combine the TNBC and normal data
combined_data = pd.concat([tnbc_data, normal_data], ignore_index=True)

# Split the data into features (X) and labels (y)
X = combined_data.drop(columns=['subtype'])
y = combined_data['subtype']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes (BL1, BL2, M, normal)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Print the classification report
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

# Function to predict subtypes for new patients
def predict_subtypes(new_data):
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    predicted_classes = predictions.argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels

# Example usage with random 10 patients from the combined dataset
# random_patients = X.sample(10, random_state=42)
# predicted_subtypes = predict_subtypes(random_patients)
# print("Predicted Subtypes for Random 10 Patients:\n", predicted_subtypes)