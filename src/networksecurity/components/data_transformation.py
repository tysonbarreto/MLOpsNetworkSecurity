from constants import training_pipeline
from entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig, DataTransformationConfig
from utils import TryExceptLogger, read_yaml_file, read_data_csv, write_yaml_file, save_numpy_array_data, save_object
from exception import NSException
from loggings import logger
from dataclasses import dataclass
import os, sys
import pandas as pd
import numpy as np
from from_root import from_root
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

logger = logger()

@dataclass
class DataTransformation:
    data_validation_artifact:DataValidationArtifact
    data_transformation_config:DataTransformationConfig

    def __post_init__(self):
        self.target_column=training_pipeline.TARGET_COLUMN
        self.kni_parameter = training_pipeline.DATA_TRANSFORMATION_IMPUTER_PARAMS

    def get_data_transformer_object(self)->Pipeline:
        imputer = KNNImputer(**self.kni_parameter)
        logger.info("KNI Imputer initialized...")
        processor=Pipeline([("imputer",imputer)])
        return processor

    def initiate_data_transformation(self)->DataTransformationArtifact:
        train_df = read_data_csv(self.data_validation_artifact.valid_train_file_path)
        test_df = read_data_csv(self.data_validation_artifact.valid_test_file_path)

        input_feature_train = train_df.drop(columns=[self.target_column], axis=1)
        target_feature_train = train_df[self.target_column]
        target_feature_train = target_feature_train.replace(-1,0)

        input_feature_test = test_df.drop(columns=[self.target_column], axis=1)
        target_feature_test = test_df[self.target_column]
        target_feature_test = target_feature_test.replace(-1,0)

        preprocessor = self.get_data_transformer_object()
        preprocessor_object = preprocessor.fit(input_feature_train)
        transformed_input_train_feature = preprocessor_object.transform(input_feature_train)
        transformed_input_test_feature = preprocessor_object.transform(input_feature_test)

        train_arr = np.c_[transformed_input_train_feature,np.array(target_feature_train)]
        test_arr = np.c_[transformed_input_test_feature,np.array(target_feature_test)]

        save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
        save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)
        save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor_object)

        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            transformed_train_file_path=self.data_transformation_config.transformed_test_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
        )
        return data_transformation_artifact
if __name__=="__main__":
    __all__ = ["DataTransformation"]










