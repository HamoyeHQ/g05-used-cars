import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.base import BaseEstimator, TransformerMixin

def testing():
    print("hello from helpers")

def model_pipeline():
    numeric_features = ['odometer']

    categorical_onehot_features = ['drive', 'fuel']
    categorical_labelenc_features = ['region', 'manufacturer', 'cylinders', 'title_status', 'transmission',
                                    'type', 'paint_color', 'state']

    

    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_onehot_transformer = Pipeline(steps=[
        ('onehot_encoder', OneHotEncoder(drop='first'))
    ])

    categorical_labelenc_transformer = Pipeline(steps=[
        ('label_encoder', OrdinalEncoder())
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ohe', categorical_onehot_transformer, categorical_onehot_features),
            ('le', categorical_labelenc_transformer, categorical_labelenc_features)
        ], remainder="passthrough")

    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor())
    ])
