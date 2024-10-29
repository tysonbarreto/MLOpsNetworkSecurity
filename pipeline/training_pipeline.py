# from src.exception import NSException
# from src.logging import logger
from src.networksecurity.components.data_ingestion import DataIngestion
from src.networksecurity.components.data_validation import DataValidation
from src.networksecurity.components.data_transformation import DataTransformation
from src.networksecurity.components.mode_trainer import ModelTrainer
from src.networksecurity.entity.config_entity import ModelTrainerConfig, DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig, DataTransformationConfig
from src.networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ClassificationMetricArtifact
from src.networksecurity.exception.NSException import NSException
from src.networksecurity.loggings.NSlogger import logger
from src.networksecurity.constants import training_pipeline
from src.networksecurity.cloud import S3Sync

from dataclasses import dataclass
import sys

logger = logger()

@dataclass
class TrainingPipeline:

    def __post_init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.training_bucket_name = training_pipeline.TRAINING_BUCKET_NAME
        self.s3_sync = S3Sync()
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("<<<<< DataIngestion successfully completed>>>>>")
            return data_ingestion_artifact
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            logger.info("<<<<< Initiating DataValidation....>>>>>")
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info("<<<<< DataValidation successfully completed>>>>>")
            return data_validation_artifact
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)
        
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation= DataTransformation(data_validation_artifact,data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info("<<<<< DataTransformation successfully completed>>>>>")
            return data_transformation_artifact
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
            model_trainer_artifact = model_trainer.initial_model_trainer()
            logger.info("<<<<< ModelTraining artifact created successfully!>>>>>")
            return model_trainer_artifact
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)
    
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{self.training_bucket_name}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
            logger.info("<<<<< Artifact folder sync'd to AWS s3 bucket >>>>>")
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)
    
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{self.training_bucket_name}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
            logger.info("<<<<< Model folder sync'd to AWS s3 bucket >>>>>")
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)
            
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logger.info('\n<<<<<Training Pipeline successfully completed>>>>>\n')
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
        except Exception as e:
            logger.info(NSException(e,sys))
            raise NSException(e,sys)
        
if __name__=="__main__":
    __all__ = ["TrainingPipeline"]