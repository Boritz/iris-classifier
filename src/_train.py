from pathlib import Path
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load the dataset
iris = load_iris()

# specify features "X" and target "y"
X = iris.data
y = iris.target

# split dataset into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create model
model = DecisionTreeClassifier(random_state=42)

# train the model using the training data
model.fit(X_train, y_train)

# predict using the test data
y_pred = model.predict(X_test)

# print both true label array and prediction data array for comparison
print("Predictions:", y_pred[:8])
print("True labels: ", y_test[:8])

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# # show how the model's predictions compare to the true labels
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", matrix)

# build path to output folder
outputs_dir = Path(__file__).resolve().parent.parent / 'outputs'
outputs_dir.mkdir(parents=True, exist_ok=True)

# save trained model to outputs folder
joblib.dump(model, outputs_dir / "model.joblib")

# create and save confusion matrix figure in outputs folder
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig(outputs_dir / "confusion_matrix.png")

# Show the confusion matrix figure
plt.show()

