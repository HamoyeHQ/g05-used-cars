import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from google.cloud import storage
import logging


def model_pipeline():
    numeric_features = ['odometer']

    # categorical_onehot_features = ['drive', 'fuel']
    categorical_labelenc_features = ['region', 'manufacturer', 'cylinders', 'size', 'condition', 
                                     'title_status','transmission', 'type', 'paint_color', 'state', 'drive', 'fuel']


    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # categorical_onehot_transformer = Pipeline(steps=[
    #     ('onehot_encoder', OneHotEncoder(drop='first')),
    # ])

    categorical_labelenc_transformer = Pipeline(steps=[
        ('label_encoder', OrdinalEncoder()),
        ('scaling', StandardScaler()),
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('le', categorical_labelenc_transformer, categorical_labelenc_features),
        ], remainder="passthrough")

    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor()),
    ])
    
    return pipeline


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to gcp bucket."""
    bucket_name = bucket_name
    source_file_name = source_file_name
    destination_blob_name = destination_blob_name

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logging.info(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )