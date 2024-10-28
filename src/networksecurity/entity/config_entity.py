from datetime import datetime
import os
from datetime import datetime
from dataclasses import dataclass, Field
from pydantic import BaseModel

from constants import training_pipeline

@dataclass
class TrainingPipelineConfig:

    def __post_init__(self):
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_name:  str = training_pipeline.ARTIFACT_DIR
        timestamp: str = datetime.now().strftime("%m_%d_%Y")
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.model_dir = os.path.join("final_model")
        
@dataclass
class DataIngestionConfig():
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        data_ingestion_dir:str=os.path.join(
            self.training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
            )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


if __name__=="__main__":
    __all__ = ["TrainingPipelineConfig","DataIngestionConfig"]
