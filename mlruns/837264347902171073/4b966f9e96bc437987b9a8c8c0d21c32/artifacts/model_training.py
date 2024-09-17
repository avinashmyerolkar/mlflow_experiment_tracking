import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import matplotlib.pyplot as plt 
import seaborn as sns

iris = load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 15
n_estimators = 10

mlflow.set_experiment("code_tracking") # if not then run will get logged in default exp

# here we will apply mlflow
with mlflow.start_run():   # alternative to mlflow.set_experiment  = experiment_id = ""
    rfc = RandomForestClassifier(max_depth=max_depth, n_estimators = n_estimators) 
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    # here we will log 2 parametres and 1 metric
    mlflow.log_metric("accuracy_score", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # track plots 
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("confusion matrics")


    # save plot as an artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    print("accuracy is :", accuracy)

    # log code artifact
    mlflow.log_artifact(__file__)

    # log the model
    mlflow.sklearn.log_model(rfc, "random_forest_classifier")

