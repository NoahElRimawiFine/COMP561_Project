import os
import pandas as pd
from Bio import SeqIO
import bisect
import json

TEST_RUNNING = False  # set to true to only sequence part of genome for faster testing

DATA_FOLDER = "datasets/"

CELL_TFBS_FILE = DATA_FOLDER + "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
PWM_FILE = DATA_FOLDER + "factorbookMotifPwm.txt"
REAL_TF_BINDING_FILE = DATA_FOLDER + "factorbookMotifPos.txt"
GENOME_DIRECTORY = DATA_FOLDER + "chromFa"
GENOME_SHAPE = DATA_FOLDER + "genome_shape"


def read_data(cell_tfbs_file, pwm_file, real_tf_binding_file, genome_shape_file):
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
    genome_shape = pd.read_csv(
        genome_shape_file, sep="\t", header=None, names=column_names
    )

    return cell_tfbs_df, pwm_dict, real_tf_binding, genome_shape


def read_genome(directory):
    sequences = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fa"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                if TEST_RUNNING:
                    for record in SeqIO.parse(file, "fasta"):
                        sequences[record.id] = str(record.seq)[:1000]
                        break
                else:
                    for record in SeqIO.parse(file, "fasta"):
                        sequences[record.id] = str(record.seq)
    return sequences


# note that there are 131 TF's in the PWM, but 133 in the real TF binding data
# this function is quite complex. Here is what it does:
# it creates a dataset of positive and negative examples. However the dataset is far too large to do this with brute force
# therefore what we do is that we first split up data by chromosome. This significantly reduces the number of examples we compare.
# secondly, we sort the positive examples. Then we only look for the positive examples in the potential neg_row that are in that thin band of being possible
# this speeds up the computation on my computer from 10+ hours to 3 minutes.
def extract_tf_examples(cell_tfbs_df, real_tf_binding_df, genome_sequences):
    positive_examples_by_chromosome = {}
    for tf in real_tf_binding_df["Transcription_Factor"].unique():
        positive_df = real_tf_binding_df[
            real_tf_binding_df["Transcription_Factor"] == tf
        ].sort_values(by="Start")
        for chrom, chrom_df in positive_df.groupby("Chromosome"):
            positive_examples_by_chromosome.setdefault(chrom, {}).setdefault(
                tf, []
            ).extend(chrom_df.to_dict("records"))

    positive_examples = {
        tf: [] for tf in real_tf_binding_df["Transcription_Factor"].unique()
    }
    negative_examples = {
        tf: [] for tf in real_tf_binding_df["Transcription_Factor"].unique()
    }

    for chrom, chrom_df in cell_tfbs_df.groupby("Chromosome"):
        if chrom in positive_examples_by_chromosome:
            for tf in negative_examples.keys():
                positive_tf_examples = positive_examples_by_chromosome[chrom].get(
                    tf, []
                )
                positive_starts = [
                    ex["Start"] for ex in positive_tf_examples
                ]  # Extract start positions

                for _, neg_row in chrom_df.iterrows():
                    start, end = neg_row["Start"], neg_row["End"]
                    is_negative = True

                    idx = bisect.bisect_left(positive_starts, start)
                    for pos_example in positive_tf_examples[idx:]:
                        if pos_example["Start"] > end:
                            break
                        if pos_example["End"] <= end:
                            is_negative = False
                            break

                    if is_negative:
                        sequence = genome_sequences[chrom][start:end]
                        negative_examples[tf].append(
                            {
                                "chromosome": chrom,
                                "start": start,
                                "end": end,
                                "sequence": sequence,
                            }
                        )
                for pos_example in positive_tf_examples:
                    sequence = genome_sequences[chrom][
                        pos_example["Start"] : pos_example["End"]
                    ]
                    positive_examples[tf].append(
                        {
                            "chromosome": chrom,
                            "start": pos_example["Start"],
                            "end": pos_example["End"],
                            "sequence": sequence,
                        }
                    )

    return positive_examples, negative_examples


def write_pos_and_neg_to_files(positive_examples, negative_examples):
    f1 = open(DATA_FOLDER + "positive_examples.json", "w")
    json.dump(positive_examples, f1)
    f2 = open(DATA_FOLDER + "negative_examples.json", "w")
    json.dump(negative_examples, f2)

    return


cell_tfbs_df, pwm_dict, real_tf_binding, genome_shape = read_data(
    CELL_TFBS_FILE, PWM_FILE, REAL_TF_BINDING_FILE, GENOME_SHAPE
)
breakpoint()
genome = read_genome(GENOME_DIRECTORY)

# to generate positive and negative files dataset
# positive_examples, negative_examples = extract_tf_examples(
#     cell_tfbs_df, real_tf_binding, genome
# )
# write_pos_and_neg_to_files(positive_examples, negative_examples)

# if you already have positive and negative files generated
