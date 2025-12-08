# DECISION TREE ON WINE DATASET (UNIQUE)

# STEP 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# STEP 2: Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Convert to DataFrame for easy understanding
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print("FIRST FIVE ROWS:")
print(df.head())

print("\nTARGET DISTRIBUTION:")
print(df['target'].value_counts())

# STEP 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Build the Decision Tree model
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    random_state=42
)

# STEP 5: Train the model
model.fit(X_train, y_train)

# STEP 6: Make predictions
y_pred = model.predict(X_test)

# STEP 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

# STEP 8: Visualize the decision tree
plt.figure(figsize=(20, 12))
plot_tree(
    model,
    filled=True,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    rounded=True
)
plt.show()
