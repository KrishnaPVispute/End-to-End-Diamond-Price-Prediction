from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import customexception
from src.logger import logging


#To save the object
from src.utils import save_object

#Data Transformation Config
###We can save the transformation model as pickle file
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')#pkl file for feature eng.




#Data Ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        "This method will create pkl file where all feature eng will donw "
        "Create complete pipeline"

        try:
            logging.info("Data Transformation Initiated")
            #Define Which col should be ordinal_encoded and which should be scaled
            categorical_cols =['cut','color','clarity']
            numerical_cols =['carat','depth','table','x','y','z']

            #Define the custom ranking for each ordinal variable
            cut_categories =['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline Initiated")

            ##Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            
            ##Categorical Pipeline
            cat_pipelines =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),#cat to num(ordinalENcoder)
                    ('scaler',StandardScaler())
                ]
            )
                #combine Both Pipeline
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipelines,categorical_cols)
            ])

            return preprocessor
        
            logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise customexception(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        "After Pipeline the transformation will initiate"

        try:
            train_df =pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data Completed")
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n {test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing Object")
            preprocessing_obj = self.get_data_transformation_object()

            #Separate the Target Column
            target_column_name ='price'  #Independent feature
            drop_columns=[target_column_name,'id']


            #Split Feature into independent and dependent feature(For Train)
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1) #Independent Feature
            target_feature_train_df =train_df[target_column_name]             #Dependent Feature


            #For test
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]



            #Apply Transformation
            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array =preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessing object on training and testing datasets.")


            #combine input_feature with target variable
            train_arr = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array,np.array(target_feature_test_df)]
    


            # TO save these pkl file we write the code in utils.py file
            save_object(
                file_path =self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Pre processor pickle is created and saved to location")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            logging.info("Exception Occured in the initiate Data Transformation")

            raise customexception(e,sys)
