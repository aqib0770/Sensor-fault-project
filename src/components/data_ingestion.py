
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os,sys
import pandas as pd
from src.utils.main_utils import MainUtils
from src.constant import *
from sklearn.model_selection import train_test_split
from pymongo.mongo_client import MongoClient
import numpy as np

@dataclass
class DataIngestionConfig:
    artifact_folder:str=os.path.join(artifact_folder)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.utils=MainUtils()

    def export_collection_as_dataframe(self,collection_name,db_name):
        try:
            # Connect to MongoDB and retrieve data from the specified collection
            mongo_client=MongoClient(MONGO_DB_URL)
            collection=mongo_client[db_name][collection_name]
            df=pd.DataFrame(list(collection.find()))
            
            # Remove the "_id" column from the DataFrame
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            
            # Replace "na" values with NaN
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_into_feature(self)->pd.DataFrame:
        try:
            logging.info("Exporting data from mongodb")
            raw_file_path=self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path,exist_ok=True)
            
            # Retrieve data from MongoDB and store it in a DataFrame
            sensor_data=self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,db_name=MONGO_DATABASE_NAME
            )
            
            logging.info(f"Saving exported data into feature_store_file_path: {raw_file_path}")
            feature_store_file_path=os.path.join(raw_file_path,'wafer_fault.csv')
            logging.info(feature_store_file_path)
            sensor_data.to_csv(feature_store_file_path,index=False)
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            feature_store_file_path=self.export_data_into_feature()
            logging.info("Data ingestion completed")
            return feature_store_file_path
        except Exception as e:
            logging.info("Error occured in data ingestion")
            raise CustomException(e,sys)
#
#In this code, the `DataIngestion` class is responsible for exporting data from MongoDB and storing it in a CSV file. The class has three main methods:
#
#1. `export_collection_as_data Thisframe`: method connects to MongoDB and retrieves data from the specified collection. It then converts the data into a pandas DataFrame and returns it.
#
#2. `export_data_into_feature`: This method calls the `export_collection_as_dataframe` method to retrieve data from MongoDB. It then stores the data in a CSV file and returns the path to the CSV file.
#
#3. `initiate_data_ingestion`: This method calls the `export_data_into_feature` method to initiate the data ingestion process. It then returns the path to the CSV file containing the exported data.
#
#The code is well-documented with comments, making it easy to understand..</s>