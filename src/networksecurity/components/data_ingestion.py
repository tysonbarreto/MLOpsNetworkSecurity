from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
from exception.NSException import NSException
from loggings.NSlogger import logger
from entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from entity.artifact_entity import DataIngestionArtifact
from utils import TryExceptLogger

import sys
import os
import pymongo
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from typing import List
import numpy as np
from pydantic import BaseModel, Field
from dataclasses import dataclass
import pandas as pd

load_dotenv()

logger = logger()
@dataclass
class DataIngestion():
    data_ingestion_config: DataIngestionConfig


    def __post_init__(self):

        MONGO_DB_UN = os.getenv('MONGO_DB_UN')
        MONGO_DB_PWD = os.getenv('MONGO_DB_PWD')
        uri = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PWD}@networksecuritycluster.bp8lc.mongodb.net/?retryWrites=true&w=majority&appName=NetworkSecurityCluster"
        self.database_name = self.data_ingestion_config.database_name
        self.collection_name = self.data_ingestion_config.collection_name
        self.mongo_client = MongoClient(uri)



    @property
    def get_dataframe(self)->pd.DataFrame:
        return self.export_collection_as_dataframe(self.database_name, self.collection_name, self.mongo_client)
    

    @staticmethod
    def export_collection_as_dataframe(db_name:str, collection_name:str, mongo_client:MongoClient)->pd.DataFrame:
        collection = mongo_client[db_name][collection_name]
        df = pd.DataFrame(list(collection.find()))
        if "id" in df.columns:
            df.drop(columns=["id"], axis=1, inplace=True)
        
        df.replace({"na":np.nan}, inplace=True)
        return df

 
    def export_collection_into_feature_store(self,dataframe:pd.DataFrame):
        feature_store_file_path=self.data_ingestion_config.feature_store_file_path
        #creating folder
        
        dir_path = os.path.dirname(feature_store_file_path)
        os.makedirs(dir_path,exist_ok=True)
        dataframe.to_csv(feature_store_file_path,index=False,header=True)
    

    def split_data_as_train_test(self,dataframe: pd.DataFrame)-> pd.DataFrame:
        train_set, test_set = train_test_split(
            dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
        )
        logger.info("<<<<Performed train test split on the dataframe>>>>>")

        logger.info("<<<<Exited split_data_as_train_test method of Data_Ingestion class>>>>>")
        
        dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        logger.info("<<<<Exporting train and test file path>>>>>")
        
        train_set.to_csv(
            self.data_ingestion_config.training_file_path, index=False, header=True
        )

        test_set.to_csv(
            self.data_ingestion_config.testing_file_path, index=False, header=True
        )
        logger.info("<<<<Exported train and test file path>>>>>")


    def initiate_data_ingestion(self)->DataIngestionArtifact:
        dataframe=self.get_dataframe
        #dataframe=self.export_collection_into_feature_store(dataframe)
        self.split_data_as_train_test(dataframe)

        dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                    test_file_path=self.data_ingestion_config.testing_file_path)
        return dataingestionartifact

if __name__=="__main__":
    __all__=["DataIngestion"]

        

