# from src.exception import NSException
# from src.logging import logger
from src.networksecurity import (NSException,logger,TryExceptLogger, DataIngestion, DataIngestionConfig,
                                 DataIngestionArtifact, TrainingPipelineConfig, training_pipeline,
                                 DataValidation, DataValidationConfig, DataTransformation, DataTransformationConfig, read_data_csv,
                                 ModelTrainer, ModelTrainerArtifact, ModelTrainerConfig)
from logging import StreamHandler, FileHandler
import logging
import sys
from pathlib import Path
from datetime import datetime
import os
from from_root import from_root, from_here

logger=logger()

def main():
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    logger.info("<<<<< DataIngestion successfully completed>>>>>")

    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation = DataValidation(data_ingestion_artifact, data_validation_config)

    logger.info("<<<<< Initiating DataValidation....>>>>>")

    data_validation_artifact = data_validation.initiate_data_validation()

    logger.info("<<<<< DataValidation successfully completed>>>>>")

    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation= DataTransformation(data_validation_artifact,data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()

    logger.info("<<<<< DataTransformation successfully completed>>>>>")

    model_trainer_config = ModelTrainerConfig(training_pipeline_config)
    model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
    model_trainer_artifact = model_trainer.initial_model_trainer()

    logger.info("<<<<< ModelTraining artifact created successfully!>>>>>")

if __name__=="__main__":
    main()

