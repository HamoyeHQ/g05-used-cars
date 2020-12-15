import sys
import logging
import argparse
import subprocess
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
                        help="hyperparameter"
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
    print("Starting...")
    args = parse_arguments()
    print(args.training_file_path)
    print(args.validation_file_path)
    train = pd.read_csv(args.training_file_path)
    validation = pd.read_csv(args.validation_file_path)
    
    # to_drop = ['id', 'year', 'latitude', 'longitude', 'model']
    # train['new_description'] = train['description']
    
    # validation['new_description'] = validation['description']
    
    # train = train[['region', 'year', 'manufacturer', 'model', 'cylinders', 'fuel', 'odometer', 'description', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state', 'price']]
    # validation = validation[['region', 'year', 'manufacturer', 'model', 'cylinders', 'fuel', 'odometer',
    #          'description', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state', 'price']]
    
    # train['newdrive'] = train.new_description.str.findall('(4wd|fwd|rwd)')
    # validation['newdrive'] = validation.new_description.str.findall('(4wd|fwd|rwd)')
    
    # train['newdrive'] = train['newdrive'].apply(', '.join)
    # validation['newdrive'] = validation['newdrive'].apply(', '.join)


    #if drive is NAN, replace with the value in 'new_drive'
    # train.loc[train['drive'].isnull(), 'drive'] = train['newdrive']
    # validation.loc[validation['drive'].isnull(), 'drive'] = validation['newdrive']
    
    
    # train['new_paint_color'] = train.new_description.str.findall(
    # '(black|blue|brown|custom|green|grey|orange|purple|red|silver|white|yellow|gray)').apply(', '.join)
    
    # validation['new_paint_color'] = validation.new_description.str.findall(
    # '(black|blue|brown|custom|green|grey|orange|purple|red|silver|white|yellow|gray)').apply(', '.join)

    # train.loc[train['paint_color'].isnull(), 'paint_color'] = train['new_paint_color']
    # validation.loc[validation['paint_color'].isnull(), 'paint_color'] = validation['new_paint_color']
    
    # train['new_manufacturer'] = train.new_description.str.findall(
    #     '(volvo|volkswagen|toyota|tesla|subaru|saturn|rover|ram|porsche|pontiac|nissan|morgan|mitsubishi|mini|mercury|mercedes-benz|mazda|lincoln|lexus|land rover|kia|jeep|jaguar|infiniti|hyundai|honda|hennessey|harley-davidson|gmc|ford|fiat|ferrari|dodge|datsun|chrysler|chevrolet|cadillac|buick|bmw|audi|aston-martin|alfa-romeo|acura)').apply(', '.join)
    
    # validation['new_manufacturer'] = validation.new_description.str.findall(
    #     '(volvo|volkswagen|toyota|tesla|subaru|saturn|rover|ram|porsche|pontiac|nissan|morgan|mitsubishi|mini|mercury|mercedes-benz|mazda|lincoln|lexus|land rover|kia|jeep|jaguar|infiniti|hyundai|honda|hennessey|harley-davidson|gmc|ford|fiat|ferrari|dodge|datsun|chrysler|chevrolet|cadillac|buick|bmw|audi|aston-martin|alfa-romeo|acura)').apply(', '.join)


    # train.loc[train['manufacturer'].isnull(), 'manufacturer'] = train['new_manufacturer']
    # validation.loc[validation['manufacturer'].isnull(), 'manufacturer'] = validation['new_manufacturer']
    
    # train = train.drop(columns=['new_description', 'newdrive',
    #                       'new_type', 'new_paint_color', 'new_manufacturer'])
    # validation = validation.drop(columns=['new_description', 'newdrive',
    #                       'new_type', 'new_paint_color', 'new_manufacturer'])
    
    
    # train = train[train['paint_color'].str.split().str.len()<2]
    # train = train[train['type'].str.split().str.len()<2]
    # train = train[train['drive'].str.split().str.len()<2]
    # train = train[train['drive'].str.split().str.len()<3]
    
    # validation = validation[validation['paint_color'].str.split().str.len() < 2]
    # validation = validation[validation['type'].str.split().str.len() < 2]
    # validation = validation[validation['drive'].str.split().str.len() < 2]
    # validation = validation[validation['drive'].str.split().str.len() < 3]

    # train = train.replace(r'^\s*$', np.nan, regex=True)
    # validation = validation.replace(r'^\s*$', np.nan, regex=True)
    
    train = train.dropna()
    validation = validation.dropna()
    # train = train[train['price'] > 0]
    # validation = validation[validation['price'] > 0]
    
    Q1 = train['price'].quantile(0.25)


    Q3 = train['price'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train[(train.price < lower_bound) |
                      (train.price > upper_bound)]
    
    train = train.drop(outliers.index)
    
    outliers = validation[(validation.price < lower_bound) |
                     (validation.price > upper_bound)]

    validation = validation.drop(outliers.index)
    
    Q1 = train['odometer'].quantile(0.25)


    Q3 = train['odometer'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train[(train.odometer < lower_bound) |
                      (train.odometer > upper_bound)]
    
    train = train.drop(outliers.index)
    
    outliers = validation[(validation.odometer < lower_bound) |
                     (validation.odometer > upper_bound)]

    validation = validation.drop(outliers.index)
    
    train.drop('description', axis=1, inplace=True)
    validation.drop('description', axis=1, inplace=True)
    
    le = LabelEncoder()
    

    train[['region', 'manufacturer', 'model', 'cylinders', 'fuel',
        'title_status', 'transmission', 'drive', 'type', 'paint_color',
        'state']] = train[['region', 'manufacturer', 'model',
                            'cylinders', 'fuel', 'title_status', 'transmission', 'drive',
                            'type', 'paint_color', 'state']].apply(le.fit_transform)
        
    validation[['region', 'manufacturer', 'model', 'cylinders', 'fuel',
           'title_status', 'transmission', 'drive', 'type', 'paint_color',
           'state']] = validation[['region', 'manufacturer', 'model',
                              'cylinders', 'fuel', 'title_status', 'transmission', 'drive',
                              'type', 'paint_color', 'state']].apply(le.transform)

    print("train\n", train.head())
    print("validation\n", validation.head())
    
    X_train = train.drop('price', axis=1).values
    y_train = train['price']
    ssc = StandardScaler()
    X_train = ssc.fit_transform(X_train.astype(float))
    
    
    X_validation = validation.drop('price', axis=1).values
    y_validation = validation['price']
    
    X_validation = sse.transform(X_validation.astype(float))
    
    regressor = ExtraTreesRegressor(
        max_depth=args.max_depth,
        max_features=args.max_features,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        n_estimators=args.n_estimators
    )
    
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_validation)
    
    r2_score = round(sklearn.metrics.r2_score(y_validation, y_pred), 3)
    rmse = round(np.sqrt(sklearn.metrics.mean_squared_error(y_validation, y_pred)), 3)
    
    logging.info('r2_score={}'.format(r2_score))
    logging.info('rmse={}'.format(rmse))
    print('score={}'.format(score))
    print('rmse={}'.format(rmse))

    if not args.hypertune:
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)

        subprocess.check_call(
            ['gsutil', 'cp', model_filename, args.gcs_path], stderr=sys.stdout)

        with open(model_filename, 'rb') as model_loaded:
            model = pickle.load(model_loaded)

        accuracy = model.score(X_validation, y_validation)
        print('Model Accuracy: {}'.format(accuracy))

        print(f'Saved model in {args.gcs_path}')


if __name__ == '__main__':
    main()
