from constants import training_pipeline
from entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig
from utils import TryExceptLogger, read_yaml_file, read_data_csv, write_yaml_file
from exception import NSException
from loggings import logger
from dataclasses import dataclass
from scipy.stats import ks_2samp
import os, sys
import pandas as pd
import numpy as np
from from_root import from_root

logger = logger()

@dataclass
class DataValidation:
    data_ingestion_artifact: DataIngestionArtifact
    #data_validation_artifact: DataValidationArtifact
    data_validation_config: DataValidationConfig
    #training_pipeline_: training_pipeline

    def __post_init__(self):
        self.schema_config = read_yaml_file(file_path=os.path.join(from_root(),training_pipeline.SCHEMA_FILE_PATH))

    def initiate_data_validation(self)->DataValidationArtifact:
        train_file_path = self.data_ingestion_artifact.trained_file_path
        test_file_path = self.data_ingestion_artifact.test_file_path

        train_dataframe = read_data_csv(train_file_path)
        test_dataframe = read_data_csv(test_file_path)

        check_status = self.validate_number_of_columns(dataframe=train_dataframe)

        data_drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
        dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)

        os.makedirs(dir_path,exist_ok=True)

        train_dataframe.to_csv(
            self.data_validation_config.valid_train_file_path, index=False, header=True
        )
        test_dataframe.to_csv(
            self.data_validation_config.valid_test_file_path, index=False, header=True
        )
        data_validation_artifact = DataValidationArtifact(
            validation_status=check_status,
            valid_train_file_path=self.data_validation_config.valid_train_file_path,
            valid_test_file_path=self.data_validation_config.valid_test_file_path,
            invalid_train_file_path=None,
            invalid_test_file_path=None,
            drift_report_file_path=self.data_validation_config.drift_report_file_path
        )
        return data_validation_artifact

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        number_of_columns=len(self.schema_config)
        logger.info(f"Required number of columns: {number_of_columns}")
        logger.info(f"DataFrame has columns: {len(dataframe.columns)}")
        if len(dataframe.columns) == number_of_columns:
            logger.info("ValidationError: DataFrame columns validated!")
            return True
        else:
            logger.info("ValidationError: DataFrame Validation has failed, due to not matching columns in data_validation.py")
            #raise Exception("ValidationError: DataFrame Validation has failed, due to not matching columns in data_validation.py")
            
    def detect_dataset_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, threshold:float=0.05):
        status = True
        report={}

        for column in base_df.columns:
            d1 = base_df[column]
            d2 = current_df[column]
            is_sample_dist = ks_2samp(d1,d2)
            if threshold<=is_sample_dist.pvalue:
                is_found=True
                status=False
            else:
                is_found=True
                status=False
            report.update({column:{
                "p_value":float(is_sample_dist.pvalue),
                "drift_status":is_found
            }})

        drift_report_file_path = self.data_validation_config.drift_report_file_path
        dir_path = os.path.dirname(drift_report_file_path)
        os.makedirs(dir_path,exist_ok=True)
        write_yaml_file(file_path=drift_report_file_path,content=report)

if __name__=="__main__":
    __all__ = ["DataValidation"]
    #print(DataValidation().schema_config)
