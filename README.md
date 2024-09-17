# MLFlow Experimentation and Model Tracking

This repository demonstrates how to track machine learning experiments using **MLflow**. It includes the configuration, logging parameters, metrics, and artifacts, model registry, and experiment tracking using MLflow's UI.

## Why is Experiment Tracking Important?
Machine learning models require multiple experiments with different hyperparameters, models, and configurations. Tracking these experiments manually (e.g., in Excel sheets) is error-prone and non-scalable. MLflow provides an efficient and automated solution to:
- Log experiments and their metadata.
- Compare different runs and track their metrics.
- Easily roll back to specific experiments that yielded better results.
  
## Tools Used
- **MLflow**: Used for experiment tracking, model registry, and serving.
- **DVC (Data Version Control)**: Used for versioning data, but less intuitive for experiment tracking than MLflow.
  
### Why MLflow Over DVC?
While DVC is still widely used for pipelines and data versioning, MLflow provides more intuitive and feature-rich tracking for experiments, thanks to:
- A comprehensive UI for comparing runs.
- Easier collaboration.
- Seamless integrations with tools like Databricks, cloud platforms, Langchain, OpenAI, and more.
  
## Major Components of MLFlow
- **MLflow Tracking**: Log parameters, metrics, and artifacts. Visualize and compare experiments via the UI.
- **Model Registry**: Manage models, including versions and metadata.
- **Model Serving**: Seamlessly serve models once they are registered.

## Experiment vs Run in MLFlow
- **Experiment**: A collection of related runs. For example, using Random Forest (RFC) and Artificial Neural Networks (ANN) as two different experiments for a classification problem.
- **Run**: Different executions within an experiment, where you might adjust hyperparameters or feature engineering steps.

## Practical Example: Getting Started

### Setup
1. **Create a new directory**:
    ```bash
    mkdir mlflow-practical
    cd mlflow-practical
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    ```

3. **Install MLFlow**:
    ```bash
    pip install mlflow
    ```

4. **Start MLFlow UI**:
    ```bash
    mlflow ui --host 0.0.0.0 --port 5000
    ```

    This will launch the MLflow UI, accessible at `http://localhost:5000`. You will see a default experiment named "Default" with its Experiment ID set to `0`. Any logged experiments or runs will be stored in the `mlruns` directory.

### Logging Runs
You can log parameters, metrics, and artifacts in MLflow using Python code. Here is an example:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set experiment
mlflow.set_experiment("my_experiment")

# Start a run
with mlflow.start_run():
    # Train your model
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    
    # Log parameters, metrics, and artifacts
    mlflow.log_param("random_state", 42)
    y_pred = rfc.predict(X_test)
    accuracy = rfc.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log confusion matrix as an artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Log model
    mlflow.sklearn.log_model(rfc, "random_forest_classifier")

    print(f"Accuracy: {accuracy}")
```

### Customizing Experiments
You can create your own experiments by specifying a name in your Python script:
```python
mlflow.set_experiment("custom_experiment_name")
```

Alternatively, you can log a run under an experiment by specifying the experiment ID:
```python
with mlflow.start_run(experiment_id="12345"):
    # log params, metrics, etc.
```

### Model Logging and Serving
MLflow supports specific model logging for frameworks like `sklearn`, `tensorflow`, `huggingface`, etc. Logging models with their specific API helps during model serving.

For example, logging an sklearn model:
```python
mlflow.sklearn.log_model(rfc, "random_forest_classifier")
```

MLflow will automatically generate code for predictions using Spark and Pandas frameworks.

