import time
from sklearn.base import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

from COMP561_Project.read_datasets import add_pwm_scores_to_data, pad_mgw_vector

def train_models_per_tf(positive_examples, negative_examples, pwm_dict):
    models_pwm_only = {}
    models_pwm_shape = {}

    max_seq_length = 0
    for tf in pwm_dict.keys():
        if (
            tf in negative_examples
            and not negative_examples[tf].empty
            and "sequence" in negative_examples[tf]
        ):
            max_seq_length = negative_examples[tf]["sequence"].str.len().max()

    for tf in pwm_dict.keys():
        # Only proceed if there are examples for the current TF
        if (
            tf in positive_examples
            and not positive_examples[tf].empty
            and tf in negative_examples
            and not negative_examples[tf].empty
        ):
            print("A")
            last_timestamp = time.time()
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

            X_train_pwm, X_test_pwm, y_train, y_test = train_test_split(
                X_pwm, y, test_size=0.3, random_state=42
            )
            X_train_shape, X_test_shape, _, _ = train_test_split(
                X_shape, y, test_size=0.3, random_state=42
            )

            print("B")
            current_time = time.time()
            print(f"Time for AB: {current_time - last_timestamp:.2f} seconds")
            last_timestamp = current_time

            breakpoint()

            # pwm model only
            model_pwm_only = RandomForestClassifier()
            model_pwm_only.fit(X_train_pwm, y_train)
            models_pwm_only[tf] = model_pwm_only

            # pwm + shape model
            model_pwm_shape = RandomForestClassifier()
            model_pwm_shape.fit(X_train_shape, y_train)
            models_pwm_shape[tf] = model_pwm_shape

            print("C")
            current_time = time.time()
            print(f"Time for BC section: {current_time - last_timestamp:.2f} seconds")
            last_timestamp = current_time

            y_pred_pwm = model_pwm_only.predict(X_test_pwm)
            y_pred_shape = model_pwm_shape.predict(X_test_shape)
            print("D")
            current_time = time.time()
            print(f"Time for CD: {current_time - last_timestamp:.2f} seconds")
            last_timestamp = current_time

            print(
                f"TF: {tf} - PWM-only model - MSE: {mean_squared_error(y_test, y_pred_pwm)}, R2: {r2_score(y_test, y_pred_pwm)}"
            )
            print(
                f"TF: {tf} - PWM + Shape model - MSE: {mean_squared_error(y_test, y_pred_shape)}, R2: {r2_score(y_test, y_pred_shape)}"
            )

    return models_pwm_only, models_pwm_shape

