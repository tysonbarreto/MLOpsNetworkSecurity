from functools import wraps
from loggings.NSlogger import logger
from exception.NSException import NSException
import sys,os
import yaml
import pandas as pd
import numpy as np
import pickle

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

def save_object(file_path:str, obj:object):

    try:        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logger.info(NSException(e,sys))

if __name__=="__main__":
    __all__=["TryExceptLogger","read_yaml_file", "read_data_csv", "write_yaml_file", "save_numpy_array_data"]