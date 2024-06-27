# Tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load the TNBC data
tnbc_file_path = 'TNBCPatientData.xlsx'
tnbc_data = pd.read_excel(tnbc_file_path, skiprows=1)

# Rename the columns based on the first row
tnbc_data.columns = ['no', 'BRCA1', 'BRCA2', 'TP53', 'EGFR (HER1)', 'MET', 'RB1', 'PIK3CA', 'subtype']
tnbc_data = tnbc_data.drop(columns=['no'])

# Load the normal data
normal_file_path = 'normalControlData.xlsx'
normal_data = pd.read_excel(normal_file_path, skiprows=1)

# Rename the columns based on the first row
normal_data.columns = ['no', 'BRCA1', 'BRCA2', 'TP53', 'EGFR (HER1)', 'MET', 'RB1', 'PIK3CA']
normal_data = normal_data.drop(columns=['no'])

# Create a label for normal data
normal_data['subtype'] = 'normal'

# Encode the labels
label_encoder = LabelEncoder()
normal_data['subtype'] = label_encoder.fit_transform(normal_data['subtype'])

# Combine normal data with TNBC data for pre-training
pretrain_data = pd.concat([tnbc_data.drop(columns=['subtype']), normal_data.drop(columns=['subtype'])])
pretrain_labels = pd.concat([pd.Series(np.zeros(len(tnbc_data))), pd.Series(np.ones(len(normal_data)))])  # 0 for TNBC, 1 for normal

# Standardize the features
scaler = StandardScaler()
pretrain_data_scaled = scaler.fit_transform(pretrain_data)

# Split pre-training dataset into training and validation sets
X_pretrain_train, X_pretrain_val, y_pretrain_train, y_pretrain_val = train_test_split(pretrain_data_scaled, pretrain_labels, test_size=0.2, random_state=42)

# Build the TensorFlow model for pre-training
pretrain_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_pretrain_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (TNBC or normal)
])

# Compile the pre-training model
pretrain_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pre-train the model
pretrain_model.fit(X_pretrain_train, y_pretrain_train, epochs=200, batch_size=16, validation_data=(X_pretrain_val, y_pretrain_val))

# Remove the last layer for fine-tuning
pretrain_model.pop()

# Add new layers for fine-tuning
pretrain_model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3 classes (BL1, BL2, M)

# Compile the model for fine-tuning
pretrain_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Encode TNBC labels
tnbc_data['subtype'] = label_encoder.fit_transform(tnbc_data['subtype'])

# Standardize the TNBC data
tnbc_data_scaled = scaler.transform(tnbc_data.drop(columns=['subtype']))

# Split the TNBC dataset into training and testing sets
X_tnbc_train, X_tnbc_test, y_tnbc_train, y_tnbc_test = train_test_split(tnbc_data_scaled, tnbc_data['subtype'], test_size=0.2, random_state=42)

# Fine-tune the model on TNBC data
pretrain_model.fit(X_tnbc_train, y_tnbc_train, epochs=200, batch_size=16, validation_split=0.1)

# Evaluate the model
y_pred = pretrain_model.predict(X_tnbc_test)
y_pred_classes = y_pred.argmax(axis=1)

# Print the classification report
report = classification_report(y_tnbc_test, y_pred_classes, target_names=label_encoder.classes_)
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