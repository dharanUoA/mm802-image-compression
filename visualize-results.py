import pandas as pd
import matplotlib.pyplot as plt

NAME_COLUMN = "Name"
NOVAL_APPROACH_SIZE_COLUMN = "Size Using Noval Approach"
JEPG_LS_APPROACH_SIZE = "Size Using JPEG-LS"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("results.csv")

filtered_df = df[df[NOVAL_APPROACH_SIZE_COLUMN] < df[JEPG_LS_APPROACH_SIZE]]

df_subset = filtered_df.head(10)

# Extract relevant columns
file_names = df_subset[NAME_COLUMN]
algo1_sizes = df_subset[NOVAL_APPROACH_SIZE_COLUMN]
algo2_sizes = df_subset[JEPG_LS_APPROACH_SIZE]

plt.figure(figsize=(12, 8))

# Set the position of the bars on the x-axis
bar_width = 0.35
index = range(len(file_names))

# Plot bars for Algorithm 1 sizes
plt.barh(index, algo1_sizes, bar_width, label='Size Using Noval Approach')

# Plot bars for Algorithm 2 sizes
plt.barh([i + bar_width for i in index], algo2_sizes, bar_width, label='Size Using JPEG-LS')

# Add labels, title, and legend
plt.xlabel('File Size')
plt.ylabel('File Name')
plt.title('Comparison of File Sizes using Both Algorithms')
plt.yticks([i + bar_width / 2 for i in index], file_names)
plt.legend()

plt.tight_layout()
plt.show()
