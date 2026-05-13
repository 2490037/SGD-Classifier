# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program

2.Data preprocessing

3.Cleanse data,handle missing values,encode categorical variables.

4.Model Training:Fit logistic regression model on preprocessed data.

5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.

6.Prediction: Predict placement status for new student data using trained model.



## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: PRANAV S
RegisterNumber:  212224040242
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add target column
df['target'] = iris.target

# Display first 5 rows
print(df.head())

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create SGD Classifier model
sgd_clf = SGDClassifier(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train the model
sgd_clf.fit(X_train, y_train)

# Predict test data
y_pred = sgd_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Oranges'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Show plot
plt.show()
```

## Output:
<img width="1166" height="503" alt="image" src="https://github.com/user-attachments/assets/9d407a4f-53bf-499a-9622-ea41134f001c" />
<img width="1132" height="729" alt="image" src="https://github.com/user-attachments/assets/9e8965c0-303e-4196-9920-3ccfb7798b68" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
