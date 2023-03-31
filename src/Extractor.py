import math
import numpy as np
import pandas as pd

#mushDF = pd.read_csv('even_smaller_data.txt', header=None)
mushDF = pd.read_csv('even_smaller_data.txt', header=None, na_values=['?'])

mushDF.columns = ['Poisonous', 'Cap', 'Stalk','Solitary', 'Fake']
print(mushDF)

mushDF = mushDF.dropna()

def oneHotEncoder(df):
    columnsToEncode = []
    for column in df.columns:
        numUniqueVelue = df[column].nunique()
        if numUniqueVelue > 2:
            columnsToEncode.append(column)
    ## get_dummies() one-hot encodes the relavent columns
    df = pd.get_dummies(df, columns= columnsToEncode)
    return df

mushDF = oneHotEncoder(mushDF)
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

mushDF = mushDF.astype(int)
print(mushDF)

mushDF = mushDF.drop('Fake', axis=1)
print(mushDF)

np.random.seed(0)
indices = np.random.permutation(mushDF.shape[0])
split_idx = int(0.8 * len(indices))
traning_mush_df = mushDF.iloc[indices[:split_idx]]
testing_mush_df = mushDF.iloc[indices[split_idx:]]

print(traning_mush_df)
print(testing_mush_df)


def my_round(x):
    EPSILON = 1e-9  
    return round(x + EPSILON)


class Adams_Node:
    def __init__(self, dataframe,feature = None, threshold=None, parent = None, left=None, right=None, depth=0):
        self.df = dataframe          # the data points in this Adams_Node 
        self.feature = feature       # the feature this Adams_Node was slit from its parent Adams_Node on
        self.threshold = threshold   # the threshold to split on (1 or 0)
        self.parent = parent         # the parent of this Adams_Node. None for the root.
        self.left = left             # the left child Adams_Node
        self.right = right           # the right child Adams_Node
        self.depth = depth           # the distance of the Adams_Node form the root
        self.prediction = my_round(self.df['Poisonous'].mean()) # the prediction if a new data point ends in this Adams_Node

root = Adams_Node(mushDF)
print('The prediction for the root node is: ',root.prediction)

def entropy_calculator(node):
    df = node.df
    n_positive = (df['Poisonous'] == 1).sum()
    n_total = df.shape[0]
    p_positive = n_positive/n_total
    p_negative = 1-p_positive 
    if (p_positive == 0 or p_positive == 1):
        return 0
    entropy = -p_positive*math.log2(p_positive) - p_negative*math.log2(p_negative)
    return entropy

print('Entropy of the root node is: ',entropy_calculator(root))

def split(node, column):
    df = node.df
    left_df = df[df[column] == 1]
    right_df = df[df[column] == 0]
    left_node = Adams_Node(dataframe = left_df, threshold = 1, parent = node, depth = node.depth+1)
    right_node = Adams_Node(dataframe = right_df, threshold = 0, parent = node, depth = node.depth+1)
    return left_node, right_node


left, right = split(root, 'Cap')
print('left df is:')
print(left.df)
print('right df is:')
print(right.df)


def fetch_best_feature(node):
    if(node == None):
      return
    df = node.df
    n_total = df.shape[0]
    curent_entropy = entropy_calculator(node)
    max_information_gain = 0
    #for debugging
    infGains  =[]
    for column in df.columns:
        #dont ever split on the response
        if column == 'Poisonous':
            continue
        left_node, right_node = split(node, column)
        # If you've already split on that feature skip
        if left_node.df.empty or right_node.df.empty:
            continue
        n_yes = left_node.df.shape[0]
        n_no = right_node.df.shape[0]
        wL = n_yes/n_total
        wR = n_no/n_total
        information_gain = curent_entropy - wL*entropy_calculator(left_node) -wR*entropy_calculator(right_node)
        infGains.append(information_gain)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = column
    node.feature = best_feature
    return best_feature 

best = fetch_best_feature(root)
print(best)

def train_tree(node, stopping_depth = 1):
    if node == None or node.depth >= stopping_depth:
      return
    feature = fetch_best_feature(node)
    left_node, right_node = split(node, feature)
    train_tree(left_node, stopping_depth)
    train_tree(right_node, stopping_depth)
    return node 

