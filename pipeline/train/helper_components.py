from typing import NamedTuple, Any
from kfp.dsl.types import GCSPath

def usedcars_model_training(
    training_file_path=None,
    validation_file_path=None,
    n_estimators=None,
    min_samples_split=None,
    min_samples_leaf=None,
    max_depth=None,
    max_features=None,
    job_dir=None,
    volume_mount=None
) -> NamedTuple('Outputs', [
    ('job_dir', str),

]):
    import os
    import subprocess
    import sys
    import pickle
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn', 'pandas', 'gcsfs', 'google-cloud-storage'])
 
    
    import numpy as np
    import pandas as pd
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
    
    pipeline = model_pipeline()
    
    train = pd.read_csv(training_file_path)
    validation = pd.read_csv(validation_file_path)
    
    pipeline.set_params(
        regressor__n_estimators=int(n_estimators),
        regressor__min_samples_split=int(min_samples_split),
        regressor__min_samples_leaf=int(min_samples_leaf),
        regressor__max_features=str(max_features),
        regressor__max_depth=int(max_depth),
    )
    
    Q1 = train['price'].quantile(0.25)
    Q3 = train['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train[(train.price < lower_bound) | (train.price > upper_bound)]
    train = train.drop(outliers.index)


    Q1 = train['odometer'].quantile(0.25)
    Q3 = train['odometer'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train[(train.odometer < lower_bound) |
                    (train.odometer > upper_bound)]

    train = train.drop(outliers.index)
    print("======================================>\n", train.iloc[3, :].values)
    X = train.drop('price', axis=1)
    y = train['price']
    pipeline.fit(X, y)
    
    X = validation.drop('price', axis=1)
    y = validation['price']

    y_pred = pipeline.predict(X)

    r2_score = round(sklearn.metrics.r2_score(y, y_pred), 3)
    rmse = round(np.sqrt(sklearn.metrics.mean_squared_error(y, y_pred)), 3)

    print('r2_score={}'.format(r2_score))
    print('rmse={}'.format(rmse))

    subprocess.run(["rm", "-r", "jobdir"])

    print("saving model...")
    model_filename = 'model.pkl'


    with open(model_filename, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    
    try:
        gcs_model_path = '{}/{}'.format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stdout=sys.stdout)
        print(f'Saved model to {gcs_model_path}')
        
    except subprocess.CalledProcessError:
        
        upload_info = job_dir.split("/", maxsplit=3)
        bucket_name = upload_info[-2]
        destination_blob_name = upload_info[-1]
        gsc_model_path = "gs/{}/{}/{}/{}".format(bucket_name, volume_mount, model_filename, destination_blob_name)
        helpers.upload_to_gcs(bucket_name, f'{volume_mount}/{model_filename}', destination_blob_name)
        print(f'Saved model to {gcs_model_path}')

    return (job_dir,)

    
    
def evaluate_model(
    dataset_path=None,
    model_path=None,
    metric_name=None,
) -> NamedTuple('Outputs', [
    ('metric_name', str),
    ('metric_value', float),
    ('pipeline_metrics', 'Metrics'),

]):
    import os
    import subprocess
    import sys
    import pickle
    import json
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas', 'gcsfs', 'google-cloud-storage'])
 
    
    import numpy as np
    import pandas as pd
    import sklearn
    import logging

    
    test = pd.read_csv(dataset_path)
    
    X = test.drop('price', axis=1)
    y = test['price']
  
    subprocess.run(["rm", "-r", "jobdir"])

    print("Loading model...")
    model_filename = 'model.pkl'
    
    gcs_model_path = '{}/{}'.format(model_path, model_filename)
    subprocess.check_call(['gsutil', 'cp', gcs_model_path, model_filename], stdout=sys.stdout)
    


    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
        print(f'Loaded model from {gcs_model_path}')
        
    y_pred = model.predict(X)
    
    if metric_name == "r2_score":
        metric_value = round(sklearn.metrics.r2_score(y, y_pred), 3)
        print('r2_score={}'.format(metric_value))
        
    elif metric_name == "rmse":   
        metric_value = round(np.sqrt(sklearn.metrics.mean_squared_error(y, y_pred)), 3)
        print('rmse={}'.format(metric_value))
        
    else:
        metric_name = 'N/A'
        metric_value = 0
        
    metrics = {
        'metrics': [{
         'name': metric_name,
         'numberValue': float(metric_value)
        }]
    }
        


    return (metric_name, metric_value, json.dumps(metrics))
    

    
def deploy_model(
    model_uri=None,
    project_id=None,
    model_id=None,
    version_id=None,
    runtime_version=None,
    python_version=None,
    replace_existing_version=None
) -> NamedTuple('Outputs', [
    ('metric_name', str),
    ('metric_value', float),
    ('pipeline_metrics', 'Metrics'),

]):
    import os
    import subprocess
    import sys
    
    print("creating model")
    try:
        subprocess.check_call(
            ["gcloud",
             'ai-platform',
             'models',
             'create',
             f'{model_id}',
             '--region=us-central1',
             ])

        print("model created")
    except subprocess.CalledProcessError:
        print("Model already exists. Moving on ...")
    
    print("creating version")
    
    subprocess.check_call(
        ["gcloud",
         'ai-platform',
         'versions',
         'create',
         f'{version_id}',
         f'--model={model_id}',
         f'--origin={model_uri}',
         f'--runtime-version={runtime_version}',
         f'--framework=scikit-learn',
         f'--python-version={python_version}',
         '--region=us-central1',
         ])
    
    print("Version created...")
    print("Deployment complete.")
    