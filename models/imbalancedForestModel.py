import pandas as pd
from imblearn.ensemble import BalancedRandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from COMP561_Project.read_datasets import add_pwm_scores_to_data, pad_mgw_vector

def train_models_per_tf(positive_examples, negative_examples, pwm_dict):
    models_pwm_only = {}
    models_pwm_shape = {}

    max_seq_length = 0
    for tf in pwm_dict.keys():
        if tf in negative_examples and not negative_examples[tf].empty:
            max_seq_length = max(max_seq_length, negative_examples[tf]["sequence"].str.len().max())

    for tf in pwm_dict.keys():
        # Proceed only if there are examples for the current TF
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
            X_shape.columns = X_shape.columns.astype(str)
            y = combined_data["label"]

            # Cross-validation setup
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            # Train PWM-only model with cross-validation
            model_pwm_only = BalancedRandomForestRegressor(n_estimators=100, random_state=42)
            cv_scores_pwm = cross_val_score(model_pwm_only, X_pwm, y, cv=kfold, scoring='r2')
            models_pwm_only[tf] = model_pwm_only

            # Train PWM + shape model with cross-validation
            model_pwm_shape = BalancedRandomForestRegressor(n_estimators=100, random_state=42)
            cv_scores_shape = cross_val_score(model_pwm_shape, X_shape, y, cv=kfold, scoring='r2')
            models_pwm_shape[tf] = model_pwm_shape

            # Print cross-validation results
            print(f"TF: {tf} - PWM-only model - Average R2: {np.mean(cv_scores_pwm)}")
            print(f"TF: {tf} - PWM + Shape model - Average R2: {np.mean(cv_scores_shape)}")

    return models_pwm_only, models_pwm_shape
