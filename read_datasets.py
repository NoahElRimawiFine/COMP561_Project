import time
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import reverse_complement
import bisect
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from COMP561_Project.translate import seq_to_shape

# TODO:
# extract negative examples using PWM
# look at +/- of binding site, use reverse compliment if necessary

TEST_RUNNING = True  # set to true to only sequence part of genome and use 1st chr only.

DATA_FOLDER = "datasets/"

CELL_TFBS_FILE = DATA_FOLDER + "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
PWM_FILE = DATA_FOLDER + "factorbookMotifPwm.txt"
REAL_TF_BINDING_FILE = DATA_FOLDER + "factorbookMotifPos.txt"
GENOME_DIRECTORY = DATA_FOLDER + "chromFa"
#GENOME_SHAPE = DATA_FOLDER + "genome_shape"


def read_data(cell_tfbs_file, pwm_file, real_tf_binding_file):
    # data for cell GM..
    column_names = ["Chromosome", "Start", "End", "Region_ID"]
    cell_tfbs_df = pd.read_csv(
        cell_tfbs_file, sep="\t", header=None, names=column_names
    )

    # data for PWM for each TF
    pwm_dict = {}
    with open(pwm_file, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            tf_name = parts[0]
            matrix = [list(map(float, row.strip(",").split(","))) for row in parts[2:]]
            pwm_dict[tf_name] = matrix

    # real transcription factor site binding data
    column_names = [
        "Chromosome",
        "Start",
        "End",
        "Transcription_Factor",
        "Score",
        "Strand",
    ]
    real_tf_binding = pd.read_csv(
        real_tf_binding_file,
        sep="\t",
        header=None,
        names=column_names,
        usecols=[1, 2, 3, 4, 5, 6],
    )

    column_names = ["Chromosome", "Start", "End", "mgw"]  # minor groove width (mgw)

    return cell_tfbs_df, pwm_dict, real_tf_binding


def reshape_genome_shape_data(genome_shape_df):
    reshaped_data = {}
    for chrom in genome_shape_df["Chromosome"].unique():
        chrom_df = genome_shape_df[genome_shape_df["Chromosome"] == chrom]
        chrom_df = chrom_df.set_index("Start")["mgw"]
        reshaped_data[chrom] = chrom_df
    return reshaped_data


def reverse_complement(seq):
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(complement[base] for base in reversed(seq.upper()))


def read_genome(directory):
    sequences = {}
    if TEST_RUNNING:  # make testing much faster by only going through one chromosome
        for filename in ["chr1.fa"]:
            if filename.endswith(".fa"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as file:
                    for record in SeqIO.parse(file, "fasta"):
                        sequences[record.id] = str(record.seq)
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".fa"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as file:
                    for record in SeqIO.parse(file, "fasta"):
                        sequences[record.id] = str(record.seq)
    return sequences


# note that there are 131 TF's in the PWM, but 133 in the real TF binding data
# this function is quite complex. Here is what it does:
# it creates a dataset of positive and negative examples. However the dataset is far too large to do this with brute force
# therefore what we do is that we first split up data by chromosome. This significantly reduces the number of examples we compare.
# secondly, we sort the positive examples. Then we only look for the positive examples in the potential neg_row that are in that thin band of being possible
# this speeds up the computation on my computer from 10+ hours to 3 minutes.
def extract_tf_examples(
    cell_tfbs_df,
    real_tf_binding_df,
    genome_sequences,
    pwm_dict,
    threshold,
):
    pwm_max = calculate_max_pwm_scores(pwm_dict)
    positive_examples_by_chromosome = {}
    for tf in real_tf_binding_df["Transcription_Factor"].unique():
        positive_df = real_tf_binding_df[
            real_tf_binding_df["Transcription_Factor"] == tf
        ].sort_values(by="Start")
        for chrom, chrom_df in positive_df.groupby("Chromosome"):
            if chrom not in positive_examples_by_chromosome:
                positive_examples_by_chromosome[chrom] = {}
            if tf not in positive_examples_by_chromosome[chrom]:
                positive_examples_by_chromosome[chrom][tf] = pd.DataFrame()
            # Concatenate the DataFrame instead of using append
            positive_examples_by_chromosome[chrom][tf] = pd.concat(
                [positive_examples_by_chromosome[chrom][tf], chrom_df],
                ignore_index=True,
            )

            if TEST_RUNNING:
                break

    positive_examples = {
        tf: pd.DataFrame() for tf in real_tf_binding_df["Transcription_Factor"].unique()
    }
    negative_examples = {
        tf: pd.DataFrame() for tf in real_tf_binding_df["Transcription_Factor"].unique()
    }

    for chrom, chrom_df in cell_tfbs_df.groupby("Chromosome"):
        if chrom in positive_examples_by_chromosome:
            tfs_to_process = negative_examples.keys()
            if TEST_RUNNING:
                tfs_to_process = list(tfs_to_process)[:5]  # First 5 TFs in test mode
            for tf in tfs_to_process:
                positive_tf_examples = positive_examples_by_chromosome[chrom].get(
                    tf, pd.DataFrame()
                )
                positive_starts = positive_tf_examples["Start"].tolist()

                for _, neg_row in chrom_df.iterrows():
                    if TEST_RUNNING and len(negative_examples[tf]) >= 50:
                        break
                    start, end = neg_row["Start"], neg_row["End"]
                    is_negative = True

                    idx = bisect.bisect_left(positive_starts, start)
                    for _, pos_example in positive_tf_examples.iloc[idx:].iterrows():
                        if pos_example["Start"] > end:
                            break
                        if pos_example["End"] <= end:
                            is_negative = False
                            break

                    if is_negative:
                        sequence = genome_sequences[chrom][start:end]
                        shape_data = seq_to_shape(
                            sequence
                        )  # Get DNA shape data using seq_to_shape

                        # Select required columns
                        selected_shape_data = shape_data[
                            ["MGW", "Roll", "ProT", "HelT"]
                        ]

                        # Check each N-length subsequence
                        pwm_length = len(next(iter(pwm_dict.values()))[0])
                        for i in range(len(sequence) - pwm_length + 1):
                            subseq = sequence[i : i + pwm_length].lower()
                            pwm_score = calculate_pwm_score(subseq, pwm_dict[tf])
                            if pwm_score >= threshold * pwm_max[tf]:
                                subseq_shape_data = selected_shape_data.iloc[
                                    i : i + pwm_length
                                ]
                                new_row = {
                                    "chromosome": chrom,
                                    "start": start + i,
                                    "end": start + i + pwm_length,
                                    "sequence": subseq,
                                    "label": 0,
                                }
                                # Include shape data as separate columns with lists
                                for col in subseq_shape_data.columns:
                                    new_row[col] = subseq_shape_data[col].tolist()

                                if "n" not in new_row["sequence"]:
                                    new_row_df = pd.DataFrame([new_row])
                                    negative_examples[tf] = pd.concat(
                                        [negative_examples[tf], new_row_df],
                                        ignore_index=True,
                                    )
                                    break

                for _, pos_example in positive_tf_examples.iterrows():
                    if TEST_RUNNING and len(positive_examples[tf]) >= 50:
                        break
                    sequence = genome_sequences[chrom][
                        pos_example["Start"] : pos_example["End"]
                    ]
                    shape_data = seq_to_shape(
                        sequence.upper()
                    )  # Get DNA shape data using seq_to_shape

                    # Select required columns
                    selected_shape_data = shape_data[["MGW", "Roll", "ProT", "HelT"]]

                    new_row = {
                        "chromosome": chrom,
                        "start": pos_example["Start"],
                        "end": pos_example["End"],
                        "sequence": sequence.lower(),
                        "label": 1,
                    }
                    # Include shape data as separate columns with lists
                    for col in selected_shape_data.columns:
                        new_row[col] = selected_shape_data[col].tolist()

                    if "n" not in new_row["sequence"]:
                        new_row_df = pd.DataFrame([new_row])
                        positive_examples[tf] = pd.concat(
                            [positive_examples[tf], new_row_df], ignore_index=True
                        )

        if TEST_RUNNING:
            break

    return positive_examples, negative_examples


def pad_shape_vector(vector, max_length, pad_value=0):
    # Replace NaN values with pad_value and pad the vector to max_length
    return [val if not pd.isna(val) else pad_value for val in vector] + [pad_value] * (
        max_length - len(vector)
    )


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

            X_train_pwm, X_test_pwm, y_train, y_test = train_test_split(
                X_pwm, y, test_size=0.3, random_state=42
            )
            X_train_shape, X_test_shape, _, _ = train_test_split(
                X_shape, y, test_size=0.3, random_state=42
            )

            # pwm model only
            model_pwm_only = LinearRegression()
            model_pwm_only.fit(X_train_pwm, y_train)
            models_pwm_only[tf] = model_pwm_only

            # pwm + shape model
            model_pwm_shape = LinearRegression()
            model_pwm_shape.fit(X_train_shape, y_train)
            models_pwm_shape[tf] = model_pwm_shape

            y_pred_pwm = model_pwm_only.predict(X_test_pwm)
            y_pred_shape = model_pwm_shape.predict(X_test_shape)

            print(
                f"TF: {tf} - PWM-only model - MSE: {mean_squared_error(y_test, y_pred_pwm)}, R2: {r2_score(y_test, y_pred_pwm)}"
            )
            print(
                f"TF: {tf} - PWM + Shape model - MSE: {mean_squared_error(y_test, y_pred_shape)}, R2: {r2_score(y_test, y_pred_shape)}"
            )
            print("----------------------------------------------------------------")

    return models_pwm_only, models_pwm_shape


def add_pwm_scores_to_data(tf, data, pwm_dict):
    data["pwm_score"] = np.zeros(len(data), dtype=float)
    for index, row in data.iterrows():
        sequence = row["sequence"]
        pwm_matrix = pwm_dict[tf]  # look into the pwm data to understand the format
        pwm_score = calculate_pwm_score(sequence, pwm_matrix)
        normalized_pwm_score = pwm_score / len(sequence) if len(sequence) > 0 else 0
        data.at[index, "pwm_score"] = normalized_pwm_score
    return data


def calculate_pwm_score(sequence, pwm_matrix):
    max_score = 0
    pwm_length = len(pwm_matrix[0])

    for i in range(len(sequence) - pwm_length + 1):
        score = 0
        for j in range(pwm_length):
            nucleotide = sequence[i + j].upper()
            if nucleotide == "A":
                score += pwm_matrix[0][j]
            elif nucleotide == "C":
                score += pwm_matrix[1][j]
            elif nucleotide == "G":
                score += pwm_matrix[2][j]
            elif nucleotide == "T":
                score += pwm_matrix[3][j]
            else:
                score += 0
        max_score = max(max_score, score)

    return max_score


def calculate_max_pwm_scores(pwm_dict):
    max_scores = {}
    for tf, pwm in pwm_dict.items():
        max_score = sum(max(column) for column in zip(*pwm))
        max_scores[tf] = max_score
    return max_scores


# read data
# cell_tfbs_df, pwm_dict, real_tf_binding = read_data(
#     CELL_TFBS_FILE, PWM_FILE, REAL_TF_BINDING_FILE
# )
# genome = read_genome(GENOME_DIRECTORY)

# # to generate positive and negative files dataset
# positive_examples, negative_examples = extract_tf_examples(
#     cell_tfbs_df, real_tf_binding, genome, pwm_dict, 0.0
# )

# # breakpoint()

# models_pwm_only, models_pwm_shape = train_models_per_tf(
#     positive_examples, negative_examples, pwm_dict
# )

# breakpoint()
