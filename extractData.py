import tarfile
import gzip
import os
import pandas as pd
import multiprocessing as mp


# first file, chromFa
file_path = '../chromFa.tar.gz'

# assuming that the files are located in the current directory
current_directory = os.getcwd()

with tarfile.open(file_path, 'r:gz') as file:
    file.extractall(current_directory + '/extracted_data')



file_path = '../factorbookMotifPos.txt.gz'

with gzip.open(file_path, 'rt') as file:
    content = file.read()

    with open(current_directory + 'extracted_data/factorbookMotifPos.txt', 'w') as new_file:
        new_file.write(content)



file_path = '../factorbookMotifPwm.txt.gz'

with gzip.open(file_path, 'rt') as file:
    content = file.read()

    with open(current_directory + '/extracted_data/factorbookMotifPwm.txt', 'w') as new_file:
        new_file.write(content)



file_path = '../wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz'

def process_chunk(chunk):
    return chunk

# Initialize multiprocessing Pool
pool = mp.Pool(mp.cpu_count())

chunk_list = []
chunksize = 10 ** 6 
with gzip.open(file_path, 'rt') as file:
    for chunk in pd.read_csv(file, sep='\t', header=None, chunksize=chunksize):
        # Asynchronously process each chunk
        processed_chunk = pool.apply_async(process_chunk, [chunk])
        chunk_list.append(processed_chunk)

# Concatenate processed chunks
processed_df = pd.concat([c.get() for c in chunk_list], ignore_index=True)

pool.close()
pool.join()

output_path = current_directory + '/extracted_data/processed_data.csv'
processed_df.to_csv(output_path, sep='\t', index=False, header=False)


