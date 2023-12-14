from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from COMP561_Project.read_datasets import add_pwm_scores_to_data, pad_mgw_vector

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

def train_models_per_tf(positive_examples, negative_examples, pwm_dict):
    models_pwm_only = {}
    models_pwm_shape = {}
    max_seq_length = max((negative_examples[tf]["sequence"].str.len().max() for tf in pwm_dict if tf in negative_examples and not negative_examples[tf].empty), default=0)

    for tf in pwm_dict.keys():
        if tf in positive_examples and not positive_examples[tf].empty and tf in negative_examples and not negative_examples[tf].empty:
            print(f"Training models for TF: {tf}")
            tf_positive = positive_examples[tf]
            tf_negative = negative_examples[tf]

            combined_data = pd.concat([tf_positive, tf_negative])
            combined_data = add_pwm_scores_to_data(tf, combined_data, pwm_dict)
            combined_data["padded_mgw_vector"] = combined_data["mgw_vector"].apply(
                lambda x: pad_mgw_vector(x, max_seq_length)
            )

            X_pwm = combined_data[["pwm_score"]]  # Features for PWM-only model
            mgw_expanded = combined_data["padded_mgw_vector"].apply(pd.Series)
            X_shape = pd.concat([combined_data[["pwm_score"]], mgw_expanded], axis=1)
            y = combined_data["label"]

            # Train, tune, and select the best PWM-only model
            best_model_pwm_only, best_score_pwm, best_params_pwm = train_and_tune_model(X_pwm, y)
            models_pwm_only[tf] = best_model_pwm_only

            # Train, tune, and select the best PWM + shape model
            best_model_pwm_shape, best_score_shape, best_params_shape = train_and_tune_model(X_shape, y)
            models_pwm_shape[tf] = best_model_pwm_shape

            # Print best scores and parameters
            print(f"TF: {tf} - Best PWM-only model R2: {best_score_pwm}, Params: {best_params_pwm}")
            print(f"TF: {tf} - Best PWM + Shape model R2: {best_score_shape}, Params: {best_params_shape}")

    return models_pwm_only, models_pwm_shape
