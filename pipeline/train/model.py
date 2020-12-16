import sys
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
import pickle
import logging
import helpers



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
    
    pipeline = helpers.model_pipeline()

    pipeline.set_params(
        regressor__n_estimators=args.n_estimators,
        regressor__min_samples_split=args.min_samples_split,
        regressor__min_samples_leaf=args.min_samples_leaf,
        regressor__max_features=args.max_features,
        regressor__max_depth=args.max_depth,
    )


    print("train\n", train.head())
    print("validation\n", validation.head())
    
    X_train = train.drop('price', axis=1)
    y_train = train['price']
    pipeline.fit(X_train, y_train)
    
    
    X_validation = validation.drop('price', axis=1)
    y_validation = validation['price']
    
    y_pred = pipeline.predict(X_validation)
    
    r2_score = round(sklearn.metrics.r2_score(y_validation, y_pred), 3)
    rmse = round(np.sqrt(sklearn.metrics.mean_squared_error(y_validation, y_pred)), 3)
    
    logging.info('r2_score={}'.format(r2_score))
    logging.info('rmse={}'.format(rmse))

    if not args.hypertune:
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)

        subprocess.check_call(
            ['gsutil', 'cp', model_filename, args.gcs_path], stderr=sys.stdout)

        # with open(model_filename, 'rb') as model_loaded:
        #     model = pickle.load(model_loaded)

        # accuracy = model.score(X_validation, y_validation)
        # print('Model Accuracy: {}'.format(accuracy))

        # print(f'Saved model in {args.gcs_path}')


if __name__ == '__main__':
    main()
