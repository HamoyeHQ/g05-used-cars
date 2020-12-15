import sys
import logging
import argparse
import subprocess
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcs-path',
                        type=str,
                        help="Local directory or GCS path to export model to"
                        )

    parser.add_argument('--training-file-path',
                        type=str,
                        required=True,
                        help="Trainset path"
                        )
    
    parser.add_argument('--validation-file-path',
                        type=str,
                        required=True,
                        help="Validation file path"
                        )
    
    
    parser.add_argument('--hypertune',
                        type=bool,
                        default=False,
                        help="Used to determine if hyperparameter tuning should be done"
                        )
    
    parser.add_argument('--n-estimators',
                        type=int,
                        default=300,
                        help="Test set path"
                        )
    
    parser.add_argument('--min-samples-split',
                        type=int,
                        default=2,
                        help="hyperparameter"
                        )
    
    parser.add_argument('--min-samples-leaf',
                        type=int,
                        default=2,
                        help="Thyperparameter"
                        )
    
    parser.add_argument('--max-features',
                        type=str,
                        default='auto',
                        help="hyperparameter"
                        )
    
    parser.add_argument('--max-depth',
                        type=int,
                        default=10,
                        help="hyperparameter"
                        )
    
    

    args, _ = parser.parse_known_args()

    return args

        

def main():
    args = parse_arguments()
    
    train = pd.read_csv(args.training_file_path)
    validation = pd.read_csv(args.validation_file_path)
    to_drop = ['id', 'year', 'latitude', 'longitude', 'model']
    
    
    
    train.drop(columns=to_drop, inplace=True)
    print("train columns: ", train.columns)
    validation.drop(columns=to_drop, inplace=True)
    train = train.dropna()
    print("train head\n", train.head())
    validation = validation.dropna()
    print("validation columns: ", validation.columns)
    print("validation head\n", validation.head())
    
    
    numeric_features = ('odometer',  'cylinders', 'price')
    le_categorical_features = ('region', 'manufacturer', 'condition',
                            'fuel', 'title_status', 'transmission', 'drive', 'car_type',
                            'paint_color', 'state')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('StandardScaling', StandardScaler(), numeric_features[0:-1]),
            ('LabelEncoding', OneHotEncoder(), le_categorical_features),
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor',  RandomForestRegressor())
    ])
    
    
    num_features_type_map = {feature: 'float64' for feature in numeric_features}
    print("feature map\n", num_features_type_map)
    
    train = train.astype(num_features_type_map)
    
    X_train = train.drop('price', axis=1)
    y_train = train['price']
    
    X_validation = validation.drop('price', axis=1)
    y_validation = validation['price']
    
    
    pipeline.set_params(
        regressor__n_estimators=args.n_estimators,
        regressor__min_samples_split=args.min_samples_split,
        regressor__min_samples_leaf=args.min_samples_leaf,
        regressor__max_features=args.max_features,
        regressor__max_depth=args.max_depth,
        
    )
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_validation, y_validation)
    logging.info('score={}'.format(score))
    print('score={}'.format(score))
    
    if not args.hypertune:
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)
            
        subprocess.check_call(['gsutil', 'cp', model_filename, args.gcs_path], stderr=sys.stdout)
    
        with open(model_filename, 'rb') as model_loaded:
            model = pickle.load(model_loaded)
        
        accuracy = model.score(X_validation, y_validation)
        print('Model Accuracy: {}'.format(accuracy))
        
        print(f'Saved model in {args.gcs_path}')
        
    
        
    


if __name__ == '__main__':
    main()

