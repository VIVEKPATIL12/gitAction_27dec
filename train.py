import argparse
import json
import joblib
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

# Select model + parameters
if args.model == "logreg":
    if args.param == "tuned":
        model = LogisticRegression(max_iter=500, solver="liblinear")
    else:
        model = LogisticRegression(max_iter=200)

elif args.model == "rf":
    if args.param == "tuned":
        model = RandomForestClassifier(n_estimators=200, max_depth=5)
    else:
        model = RandomForestClassifier(n_estimators=100)

elif args.model == "svm":
    if args.param == "tuned":
        model = SVC(kernel="rbf", C=2)
    else:
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
metrics_file = f"metrics_{args.model}_{args.param}.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

# Save model with descriptive name
model_file = f"model_{args.model}_{args.param}.pkl"
joblib.dump(model, model_file)

print(f"✅ Training complete for {args.model} ({args.param})")
print(f"📊 Metrics saved to {metrics_file}")
print(f"💾 Model saved to {model_file}")
