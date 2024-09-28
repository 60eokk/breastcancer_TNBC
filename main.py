# Tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# AWS imports
import boto3
import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel

# AWS Setup
boto_session = boto3.Session(region_name='us-east-1')  
sagemaker_session = sagemaker.Session(boto_session=boto_session)
s3 = boto3.resource('s3')

# S3 bucket configuration
bucket = 'breastcancer-tnbc'
prefix = 'tnbc-classification'

def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = f"{prefix}/{file_name}"
    try:
        s3.Bucket(bucket).upload_file(Filename=file_name, Key=object_name)
    except Exception as e:
        print(f"Error uploading {file_name} to S3: {e}")
        return False
    return True

# Upload data files to S3
for file in ['control data (Normal).xlsx', 'TNBCPatientData.xlsx', 'normalControlData.xlsx', 'control data (TNBC).xlsx']:
    upload_to_s3(file, bucket)

def check_tnbc_suspective(patient_data):
    if patient_data['FOXA1'] <= 1320.1436:
        if patient_data['C6orf146'] <= 0:
            if patient_data['GNAO1'] > 173.0652:
                return 'normal'
            else:
                if patient_data['GAGE13'] <= 0:
                    if patient_data['SMCP'] > 0.3651:
                        return 'normal'
                    else:
                        if patient_data['GALP'] > 0.3808:
                            return 'normal'
                        else:
                            if patient_data['CDK17'] <= 804.2082:
                                if patient_data['AQR'] <= 455.7455:
                                    return 'normal'
                                else:
                                    if patient_data['OR10H2'] <= 0:
                                        if patient_data['C13orf34'] <= 126.3544:
                                            return 'normal'
                                        else:
                                            if patient_data['HIST1H4L'] > 0.815:
                                                return 'normal'
                                            else:
                                                if patient_data['CHST8'] > 293.0429:
                                                    return 'normal'
                                                else:
                                                    if patient_data['FABP5'] <= 85.6336:
                                                        if patient_data['HPR'] > 0.2361:
                                                            return 'TNBC'
                                                        else:
                                                            return 'normal'
                                                    else:
                                                        if patient_data['SRDSA1'] <= 1866.936:
                                                            return 'TNBC'
                                                        else:
                                                            if patient_data['391714'] > 0:
                                                                return 'TNBC'
                                                            else:
                                                                return 'normal'
                                    else:
                                        return 'normal'
                                    
                            else:
                                return 'normal'
                
                else:
                    return 'TNBC'
        else:
            return 'TNBC'
    else:
        if patient_data['ZP2'] > 41.4533:
            if patient_data['CTRC'] <= 0.3677:
                if patient_data['C1orf110'] <= 0.4803:
                    return 'normal'
                else:
                    return 'TNBC'
            else:
                return 'TNBC'
                if patient_data['TCP10L2'] > 0.2695:
                    return 'normal'
                else:
                    return 'TNBC'
        else:
            if patient_data['TCP10L2'] > 0.2695:
                if patient_data['FAM120AOS'] > 1121.6393:
                    return 'normal'
                else:
                    return 'TNBC'
            else:
                if patient_data['GABRR2'] <= 5.1181:
                    return 'normal'
                else:
                    if patient_data['GPR88'] > 17.5683:
                        return 'TNBC'
                    else:
                        return 'normal'


# Example usage with normal data
normal_data_file_path = 'control data (Normal).xlsx'
normal_data = pd.read_excel(normal_data_file_path)

results = ""
for index, row in normal_data.iterrows():
    result = check_tnbc_suspective(row)
    results += f"no.{row['no.']} : {result}\n"

print(results)


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
    predictions = pretrain_model.predict(new_data_scaled)
    predicted_classes = predictions.argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels

def predict_subtypes(test_file_path):
    # Load the test data
    test_data = pd.read_excel(test_file_path)
    
    # Extract the 'no' column for result formatting
    no_column = test_data['no.']
    
    # Drop the 'no' column and keep only the first 7 columns for prediction
    test_data = test_data.iloc[:, 1:8]  # This keeps only the first 7 columns after the 'no.' column
    
    # Standardize the features using the same scaler used for training
    test_data_scaled = scaler.transform(test_data)
    
    # Make predictions
    predictions = pretrain_model.predict(test_data_scaled[:70])
    predicted_classes = predictions.argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    
    # Format the results
    results = ""
    for no, subtype in zip(no_column[:70], predicted_labels):
        results += f"no.{no} : {subtype}\n"
    return results

# Example usage
test_file_path = 'control data (TNBC).xlsx'
predicted_results = predict_subtypes(test_file_path)
print(predicted_results)

# AWS Integration: Save and deploy the model
def save_and_deploy_model():
    # Save the model locally
    pretrain_model.save('local_model')
    
    # Upload the model to S3
    upload_to_s3('local_model', bucket, f"{prefix}/model")
    
    # Deploy the model to SageMaker
    tensorflow_model = TensorFlowModel(model_data=f"s3://{bucket}/{prefix}/model",
                                       role='SageMakerRole',  # Replace with your SageMaker role ARN
                                       framework_version='2.6')
    predictor = tensorflow_model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
    
    return predictor

# AWS Integration: Predict using the deployed model
def predict_with_sagemaker(predictor, test_file_path):
    # Load and preprocess the test data (similar to the original predict_subtypes function)
    test_data = pd.read_excel(test_file_path)
    no_column = test_data['no.']
    test_data = test_data.iloc[:, 1:8]
    test_data_scaled = scaler.transform(test_data)
    
    # Make predictions using the SageMaker endpoint
    predictions = predictor.predict(test_data_scaled[:70])
    predicted_classes = predictions['predictions'].argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    
    # Format the results
    results = ""
    for no, subtype in zip(no_column[:70], predicted_labels):
        results += f"no.{no} : {subtype}\n"
    return results

# Main execution with AWS integration
if __name__ == "__main__":
    # Run the original prediction
    print("Original Prediction Results:")
    print(predicted_results)
    
    # Save and deploy the model to SageMaker
    print("\nDeploying model to SageMaker...")
    sagemaker_predictor = save_and_deploy_model()
    
    # Make predictions using the deployed SageMaker model
    print("\nSageMaker Prediction Results:")
    sagemaker_results = predict_with_sagemaker(sagemaker_predictor, test_file_path)
    print(sagemaker_results)
    
    # Clean up: delete the SageMaker endpoint
    print("\nCleaning up SageMaker endpoint...")
    sagemaker_predictor.delete_endpoint()

    # AWS Configuration
    # Copyexport AWS_ACCESS_KEY_ID=
    # export AWS_SECRET_ACCESS_KEY=
    # export AWS_DEFAULT_REGION=