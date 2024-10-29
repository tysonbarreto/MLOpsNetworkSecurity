from utils import load_object, load_numpy_array_data
from constants import training_pipeline
from entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from loggings import logger
from utils import evaluate_models, get_classification_score, NetworkModel, write_yaml_file, save_object
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
import os, sys
import pandas as pd
import numpy as np
import pickle
from typing import Any
import mlflow
import dagshub


logger = logger()

@dataclass
class ModelTrainer:
    data_transformation_artifact: DataTransformationArtifact
    model_trainer_config: ModelTrainerConfig

    def __post_init__(self):
        self.preprocessor_path = self.data_transformation_artifact.transformed_object_file_path
        self.model_path = self.model_trainer_config.trained_model_file_path
        self.train_file_path = self.data_transformation_artifact.transformed_train_file_path
        self.test_file_path = self.data_transformation_artifact.transformed_test_file_path

    def track_mlflow(self, best_model:Any, classification_score_metrics:ClassificationMetricArtifact):
        dagshub.init(repo_owner='tysonbarreto', repo_name='MLOpsNetworkSecurity', mlflow=True)
        with mlflow.start_run():
            f1_score = classification_score_metrics.f1_score
            precision_score = classification_score_metrics.precision_score
            recall_score = classification_score_metrics.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    
    def train_model(self,X_train, y_train, X_test, y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        y_train_pred=best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        

        #Tracking with MLFlow
        self.track_mlflow(best_model=best_model, classification_score_metrics=classification_train_metric)

        preprocessor = load_object(file_path=self.preprocessor_path)
        model_dir_path = os.path.dirname(self.model_path)
        os.makedirs(model_dir_path,exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_path, obj=network_model)

        save_object("final_model/model.pkl",best_model)
        save_object("final_model/preprocessor.pkl",preprocessor)


        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logger.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    def initial_model_trainer(self)->ModelTrainerArtifact:
        train_arr = load_numpy_array_data(self.train_file_path)
        test_arr = load_numpy_array_data(self.test_file_path)

        X_train, y_train, X_test, y_test = (
            train_arr[:,:-1],
            train_arr[:,-1],

            test_arr[:,:-1],
            test_arr[:,-1]
        )

        model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
        return model_trainer_artifact
    
if __name__=="__main__":
    __all__=["ModelTrainer"]
    


