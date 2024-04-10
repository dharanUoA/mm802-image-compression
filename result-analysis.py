import pandas as pd

NAME_COLUMN = "Name"
NOVAL_APPROACH_SIZE_COLUMN = "Size Using Noval Approach"
JEPG_LS_APPROACH_SIZE = "Size Using JPEG-LS"

df = pd.read_csv("results-without-huffman.csv")

difference = df[NOVAL_APPROACH_SIZE_COLUMN] - df[JEPG_LS_APPROACH_SIZE]
print(difference.sum()/(1024*len(df)))