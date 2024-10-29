from functools import wraps
from loggings.NSlogger import logger
from exception.NSException import NSException
from constants import training_pipeline
import sys,os
import yaml
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass

from entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Any



logger = logger()

def TryExceptLogger(func):
    @wraps(func)
    def _wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            logger().info(NSException(e,sys))
    return _wrapper

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path) as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logger().info(NSException(e,sys))


def read_data_csv(file_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.info(NSException(e,sys))    

def write_yaml_file(file_path:str, content:object, replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            yaml.dump(content,file)
    except Exception as e:
        logger.info(NSException(e,sys))    

def save_numpy_array_data(file_path:str, array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file=file_obj,arr=array)
    except Exception as e:
        logger.info(NSException(e,sys))

def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        logger.info(NSException(e,sys))

def save_object(file_path:str, obj:object):

    try:        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logger.info(NSException(e,sys))

def load_object(file_path:str)->object:

    try:        
        if not os.path.exists(file_path):
            logger.info(FileNotFoundError(f"The file: {file_path} is not exists"))
            raise FileNotFoundError(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.info(NSException(e,sys))

def evaluate_models(X_train, y_train,X_test,y_test,models,param)->dict:
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logger.info(NSException(e,sys))

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        logger.info(NSException(e,sys))

@dataclass
class NetworkModel:
    preprocessor:Pipeline # comes from data transformation
    model: Any #comes from model trainer
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            logger.info(NSException(e,sys))

if __name__=="__main__":
    __all__=["TryExceptLogger","read_yaml_file", "read_data_csv", "write_yaml_file", "save_numpy_array_data","load_numpy_array_data","load_object",
             "evaluate_models","get_classification_score","NetworkModel"]