# new datasets to data mine!
# Random Forest Classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)