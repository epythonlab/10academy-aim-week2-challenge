import pickle
import mlflow
import mlflow.sklearn

# Log model
mlflow.sklearn.log_model(model, "model")

# Log parameters and metrics
mlflow.log_param("param1", value1)
mlflow.log_metric("metric1", value1)


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
