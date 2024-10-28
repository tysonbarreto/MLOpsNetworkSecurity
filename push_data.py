
import os
import sys
import json
import pandas as pd
from src.networksecurity import NSException, logger
import pymongo
from from_root import from_root

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_UN = os.getenv('MONGO_DB_UN')
MONGO_DB_PWD = os.getenv('MONGO_DB_PWD')


class NetworkDataExtract:
    def __init__(self):
        self.uri:str = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PWD}@networksecuritycluster.bp8lc.mongodb.net/?retryWrites=true&w=majority&appName=NetworkSecurityCluster"

    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NSException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(self.uri)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NSException(e,sys)
        
if __name__=='__main__':
    FILE_PATH=from_root("artifacts\phisingData.csv")
    DATABASE="NetworkSecurityDB"
    Collection="NetworkSecurity"
    networkobj=NetworkDataExtract()
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)

    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
