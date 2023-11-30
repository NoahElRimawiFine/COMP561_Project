import os
import pandas as pd
from Bio import SeqIO

TEST_RUNNING = False  # set to true to only sequence part of genome for faster testing

DATA_FOLDER = "datasets/"

CELL_TFBS_FILE = DATA_FOLDER + "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
PWM_FILE = DATA_FOLDER + "factorbookMotifPwm.txt"
REAL_TF_BINDING_FILE = DATA_FOLDER + "factorbookMotifPos.txt"
GENOME_DIRECTORY = DATA_FOLDER + "chromFa"


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

    return cell_tfbs_df, pwm_dict, real_tf_binding


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
def extract_tf_examples(cell_tfbs_df, real_tf_binding_df, genome_sequences):
    positive_examples = {}
    negative_examples = {}
    for tf in real_tf_binding_df["Transcription_Factor"].unique():
        print(f"extracting data for {tf}")
        positive_df = real_tf_binding_df[
            real_tf_binding_df["Transcription_Factor"] == tf
        ]
        positive_examples[tf] = [
            {
                "chromosome": row["Chromosome"],
                "start": row["Start"],
                "end": row["End"],
                "sequence": genome_sequences[row["Chromosome"]][
                    row["Start"] : row["End"]
                ],
            }
            for _, row in positive_df.iterrows()
        ]

        # TODO: I'm uncertain about what counts as a "sequence" in the region.
        # Is it every possible sequence there?
        # Or is the whole region the sequence?
        negative_regions = cell_tfbs_df[
            ~cell_tfbs_df.apply(
                lambda row: any(
                    (positive_df["Chromosome"] == row["Chromosome"])
                    & (positive_df["Start"] <= row["End"])
                    & (positive_df["End"] >= row["Start"])
                ),
                axis=1,
            )
        ]

        negative_examples[tf] = [
            {
                "chromosome": row["Chromosome"],
                "start": row["Start"],
                "end": row["End"],
                "sequence": genome_sequences[row["Chromosome"]][
                    row["Start"] : row["End"]
                ],
            }
            for _, row in negative_regions.iterrows()
        ]

    return positive_examples, negative_examples


cell_tfbs_df, pwm_dict, real_tf_binding = read_data(
    CELL_TFBS_FILE, PWM_FILE, REAL_TF_BINDING_FILE
)
genome = read_genome(GENOME_DIRECTORY)

positive_examples, negative_examples = extract_tf_examples(
    cell_tfbs_df, real_tf_binding, genome
)
