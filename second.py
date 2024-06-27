# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report

# # Load the provided Excel file
# file_path = 'newData.xlsx'
# data = pd.read_excel(file_path, skiprows=1)

# # Rename the columns based on the first row
# data.columns = ['no', 'BRCA1', 'BRCA2', 'TP53', 'EGFR (HER1)', 'MET', 'RB1', 'PIK3CA', 'subtype']

# # Drop the 'no' column as it is unnecessary
# data = data.drop(columns=['no'])

# # Split the data into features (X) and labels (y)
# X = data.drop(columns=['subtype'])
# y = data['subtype']

# # Encode the labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the dataset into training (80%) and testing (20%) sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # Build the TensorFlow model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# # Evaluate the model
# y_pred = model.predict(X_test)
# y_pred_classes = y_pred.argmax(axis=1)

# # Print the classification report
# report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
# print("Classification Report:\n", report)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load the provided Excel file
file_path = 'newData.xlsx'
data = pd.read_excel(file_path, skiprows=1)

# Rename the columns based on the first row
data.columns = ['no', 'BRCA1', 'BRCA2', 'TP53', 'EGFR (HER1)', 'MET', 'RB1', 'PIK3CA', 'subtype']

# Drop the 'no' column as it is unnecessary
data = data.drop(columns=['no'])

# Split the data into features (X) and labels (y)
X = data.drop(columns=['subtype'])
y = data['subtype']

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
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Print the classification report
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
print("Classification Report:\n", report)