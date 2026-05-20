# import mlflow
# import pandas as pd
# import mlflow.xgboost
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score

# def train_model(df: pd.DataFrame, target_col: str):
#     """
#     Trains an XGBoost model and logs with MLflow.

#     Args:
#         df (pd.DataFrame): Feature dataset.
#         target_col (str): Name of the target column.
#     """
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = XGBClassifier(
#         n_estimators=300,
#         learning_rate=0.1,
#         max_depth=6,
#         random_state=42,
#         n_jobs=-1,
#         eval_metric="logloss"
#     )
#     # After fitting your model, before mlflow.sklearn.log_model(...)
#     feature_columns = list(X_train.columns)  # exact columns after preprocessing

#     with open("feature_columns.txt", "w") as f:
#         f.write("\n".join(feature_columns))

#     mlflow.log_artifact("feature_columns.txt", artifact_path="model")
    
#     with mlflow.start_run():
#         # Train model
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         rec = recall_score(y_test, preds)

#         # Log params, metrics, and model
#         mlflow.log_param("n_estimators", 300)
#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_metric("recall", rec)
#         mlflow.xgboost.log_model(model, "model")

#         # 🔑 Log dataset so it shows in MLflow UI
#         train_ds = mlflow.data.from_pandas(df, source="training_data")
#         mlflow.log_input(train_ds, context="training")

#         print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}"import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


def train_model(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Save and log feature columns inside the same run
        feature_columns = list(X_train.columns)
        with open("feature_columns.txt", "w") as f:
            f.write("\n".join(feature_columns))
        mlflow.log_artifact("feature_columns.txt", artifact_path="model")

        # Log dataset
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"✅ Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
        print(f"✅ Feature columns saved: {len(feature_columns)} columns")