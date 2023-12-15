from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import pandas as pd
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from COMP561_Project.translate import seq_to_shape
from COMP561_Project.read_datasets import read_data, read_genome, extract_tf_examples, add_pwm_scores_to_data, pad_shape_vector

# read data
DATA_FOLDER = "datasets/"

CELL_TFBS_FILE = DATA_FOLDER + "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
PWM_FILE = DATA_FOLDER + "factorbookMotifPwm.txt"
REAL_TF_BINDING_FILE = DATA_FOLDER + "factorbookMotifPos.txt"
GENOME_DIRECTORY = DATA_FOLDER + "chromFa"

cell_tfbs_df, pwm_dict, real_tf_binding = read_data(
    CELL_TFBS_FILE, PWM_FILE, REAL_TF_BINDING_FILE
)
genome = read_genome(GENOME_DIRECTORY)

# to generate positive and negative files dataset
positive_examples, negative_examples = extract_tf_examples(
    cell_tfbs_df, real_tf_binding, genome, pwm_dict, 0.0
)

def train_and_tune_model(X, y, n_splits=5):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5]
    }

    model = GradientBoostingRegressor(random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Setup the grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='r2')
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    return best_model, best_score, best_params

def train_models_per_tf(positive_examples, negative_examples, pwm_dict, train_and_tune_model):
    models_pwm_only = {}
    models_pwm_shape = {}

    max_seq_length = 0
    for tf in pwm_dict.keys():
        if tf in negative_examples and not negative_examples[tf].empty and "sequence" in negative_examples[tf]:
            max_seq_length = max(max_seq_length, negative_examples[tf]["sequence"].str.len().max())

    for tf in pwm_dict.keys():
        # Only proceed if there are examples for the current TF
        if tf in positive_examples and not positive_examples[tf].empty and tf in negative_examples and not negative_examples[tf].empty:
            tf_positive = positive_examples[tf]
            tf_negative = negative_examples[tf]

            combined_data = pd.concat([tf_positive, tf_negative])
            combined_data = add_pwm_scores_to_data(tf, combined_data, pwm_dict)

            # Initialize list to collect all expanded shape feature columns
            expanded_shape_columns = []

            # Pad and expand shape features
            for feature in ["MGW", "Roll", "ProT", "HelT"]:
                combined_data[f"padded_{feature}_vector"] = combined_data[
                    feature
                ].apply(lambda x: pad_shape_vector(x, max_seq_length))
                expanded_feature = combined_data[f"padded_{feature}_vector"].apply(
                    pd.Series
                )
                expanded_feature.columns = [
                    f"{feature}_{i}" for i in expanded_feature.columns
                ]
                expanded_shape_columns.extend(
                    expanded_feature.columns
                )  # Add expanded columns to the list
                combined_data = pd.concat([combined_data, expanded_feature], axis=1)

            X_pwm = combined_data[["pwm_score"]]  # Features for PWM-only model

            # Use the list of expanded shape feature columns for the shape model
            X_shape = pd.concat(
                [combined_data[["pwm_score"]], combined_data[expanded_shape_columns]],
                axis=1,
            )

            y = combined_data["label"]

            # Train and tune the PWM-only model
            best_model_pwm_only, best_score_pwm, best_params_pwm = train_and_tune_model(X_pwm, y)
            models_pwm_only[tf] = best_model_pwm_only

            # Train and tune the PWM + shape model
            best_model_pwm_shape, best_score_shape, best_params_shape = train_and_tune_model(X_shape, y)
            models_pwm_shape[tf] = best_model_pwm_shape

            # Print performance metrics and best parameters
            print(
                f"TF: {tf} - Best PWM-only model R2: {best_score_pwm}, Params: {best_params_pwm}"
            )
            print(
                f"TF: {tf} - Best PWM + Shape model R2: {best_score_shape}, Params: {best_params_shape}"
            )
            print("----------------------------------------------------------------")

    return models_pwm_only, models_pwm_shape


models_pwm_only, models_pwm_shape = train_models_per_tf(
    positive_examples, negative_examples, pwm_dict, train_and_tune_model)