import pandas as pd

# Read the data from CSV file and replace ? with NaN
df = pd.read_csv('even_smaller_data.txt', header=None, na_values=['?'])

# Rename columns
df.columns = ['Edible', 'Cap', 'Stalk','Solitary']

# Print the DataFrame
print(df)

# Drop the rows with 'NaN'
df = df.dropna()

# Define the columns that have more than two possible values
cols_to_encode = []
for col in df.columns:
    num_unique = df[col].nunique()
    if num_unique > 2:
        cols_to_encode.append(col)

# Use pandas' get_dummies function to one-hot encode the columns
df = pd.get_dummies(df, columns=cols_to_encode)

# Print the encoded data frame
print(df)

def make_mapping_dict(df):
    mapping_dict = {}
    for col in df.columns:
        unique_vals = df[col].unique()
        if set(unique_vals) == set([0, 1]):
            # If both 1 and 0 are present, leave the column unchanged
            continue
        if len(unique_vals) != 2:
            # If there are more than two unique values, skip the column
            continue
        val_dict = dict(zip(unique_vals, range(len(unique_vals))))
        # Find the character that should be mapped to 1
        one_char = max(val_dict, key=val_dict.get)
        # Find the character that should be mapped to 0
        zero_char = min(val_dict, key=val_dict.get)
        mapping_dict[one_char] = 1
        mapping_dict[zero_char] = 0
    return mapping_dict

# Replace characters with 0 and 1
mapping_dict = make_mapping_dict(df)
df = df.replace(mapping_dict)

# Convert data type to integer
df = df.astype(int)

print(df)