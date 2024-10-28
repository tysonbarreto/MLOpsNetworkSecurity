# from src.exception import NSException
# from src.logging import logger
from src.networksecurity import NSException,logger,TryExceptLogger, DataIngestion, DataIngestionConfig,DataIngestionArtifact, TrainingPipelineConfig, training_pipeline
from logging import StreamHandler, FileHandler
import logging
import sys
from pathlib import Path
from datetime import datetime
import os
from from_root import from_root, from_here
# load_dotenv()



# MONGO_DB_UN = os.getenv('MONGO_DB_UN')
# MONGO_DB_PWD = os.getenv('MONGO_DB_PWD')


# def DBClient():
#     uri = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PWD}@networksecuritycluster.bp8lc.mongodb.net/?retryWrites=true&w=majority&appName=NetworkSecurityCluster"
#     # Create a new client and connect to the server
#     client = MongoClient(uri)
#     # Send a ping to confirm a successful connection
#     try:
#         client.admin.command('ping')
#         print("Pinged your deployment. You successfully connected to MongoDB!")
#     except Exception as e:
#         logger.info(NSException(e,sys))

# if __name__=="__main__":
#     __all__=["DBClient"]


def main()->DataIngestionArtifact:
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    print(data_ingestion_artifact)


if __name__=="__main__":
    main()

