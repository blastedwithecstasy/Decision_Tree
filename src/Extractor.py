import pandas as pd

## Read the data from CSV file and replace '?' with 'NaN'
mushDF = pd.read_csv('even_smaller_data.txt', header=None, na_values=['?'])

## Rename columns
mushDF.columns = ['Poisonous', 'Cap', 'Stalk','Solitary', 'Fake']
print(mushDF)

## Delete the data points with missing values 
mushDF = mushDF.dropna()

def oneHotEncoder(df):
    ## Find the attributes with more than two possible values 
    columnsToEncode = []
    for column in df.columns:
        numUniqueVelue = df[column].nunique()
        if numUniqueVelue > 2:
            columnsToEncode.append(column)
    ## get_dummies() one-hot encodes the relavent columns
    df = pd.get_dummies(df, columns= columnsToEncode)
    return df

mushdf = oneHotEncoder(mushDF)
print(mushDF)

## Replaces the letters with '1's and '0's
def replaceWithBinary(df):
    for col in df.columns:
        uniqueVals = df[col].unique()
        if set(uniqueVals) == set([0, 1]):
            # If both 1 and 0 are present, leave the column unchanged
            continue
        if len(uniqueVals) != 2:
            # If there are more than two unique values, skip the column
            continue
        ## Create a dictionary mapping that maps the max letter to 1 and the min letter to 0
        mappingDict = {max(uniqueVals): 1, min(uniqueVals): 0}
        df[col] = df[col].replace(mappingDict)
    return df

mushDF = replaceWithBinary(mushDF) 

## Convert data type of the data frame to integer
mushDF = mushDF.astype(int)
print(mushDF)

mushDF = mushDF.drop('Fake', axis=1)
print(mushDF)