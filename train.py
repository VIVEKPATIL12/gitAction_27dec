import argparse
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf", "svm"])
parser.add_argument("--param", type=str, default="default")
args = parser.parse_args()

# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select model
if args.model == "logreg":
    model = LogisticRegression(max_iter=200)
elif args.model == "rf":
    model = RandomForestClassifier(n_estimators=100)
elif args.model == "svm":
    model = SVC(kernel="linear")

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
metrics = {
    "model": args.model,
    "param": args.param,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="macro"),
    "recall": recall_score(y_test, y_pred, average="macro")
}

# Save metrics
filename = f"metrics_{args.model}_{args.param}.json"
with open(filename, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Training complete for {args.model} ({args.param}). Metrics saved to {filename}")
# End of train.py