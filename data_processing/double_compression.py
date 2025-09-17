import io
import pandas as pd
import time
import os

# Load your CSV once
csv_path = '../data/race_stats.csv'
feather_path1 = '../data/race_stats_sca.feather'
feather_path2 = '../data/race_stats_vec.feather'
names = [feather_path1, feather_path2]

df = pd.read_csv(csv_path)

# Read the raw file first

with open(csv_path, "r") as f:

    lines = f.readlines()



# Detect header rows (assuming they all start the same way as the first header)

header = lines[0].strip()

chunks = []

current_chunk = []



for line in lines:

    if line.strip() == header:  # found a new header

        if current_chunk:       # save previous chunk

            chunks.append(current_chunk)

        current_chunk = [line]  # start new chunk with the header

    else:

        current_chunk.append(line)



# Save last chunk

if current_chunk:

    chunks.append(current_chunk)



# Parse each chunk into its own DataFrame

dfs = [pd.read_csv(io.StringIO("".join(chunk))) for chunk in chunks]

for df, name in zip(dfs, names):

# Test Feather
	start = time.time()
	df.to_feather(name)
	feather_write_time = time.time() - start

	start = time.time()
	df_feather = pd.read_feather(name)
	feather_read_time = time.time() - start

# Check file sizes
	csv_size = os.path.getsize(csv_path) / (1024*1024)  # MB
	feather_size = os.path.getsize(name) / (1024*1024)  # MB

	print(f"CSV size: {csv_size:.1f}MB")
	print(f"Feather size: {feather_size:.1f}MB")
	print(f"Feather read time: {feather_read_time:.2f}s")
	print(f"Feather write time: {feather_write_time:.2f}s")

	print(df_feather.head())
