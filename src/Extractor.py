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
        self.df = dataframe          
        self.feature = feature       
        self.threshold = threshold   
        self.parent = parent         
        self.left = left             
        self.right = right           
        self.depth = depth          
        self.prediction = my_round(self.df['Poisonous'].mean()) 

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
    if column == None:
        return
    df = node.df
    left_df = df[df[column] == 1]
    right_df = df[df[column] == 0]
    if right_df.shape[0] == 0 or left_df.shape[0] == 0:
        return
    left_node = Adams_Node(dataframe = left_df, threshold = 1, parent = node, depth = node.depth+1)
    right_node = Adams_Node(dataframe = right_df, threshold = 0, parent = node, depth = node.depth+1)
    node.left = left_node
    node.right = right_node
    return left_node, right_node


def fetch_best_feature(node):
    if(node == None):
      return
    best_feature = None
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
        split_result = split(node, column)
        # If you've already split on that feature skip
        if split_result is None:
            continue
        left_node, right_node = split_result
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


def train_tree(node, stopping_depth=1):
    if node.depth >= stopping_depth:
        return
    feature = fetch_best_feature(node)
    split_result = split(node, feature)
    if split_result is None or feature == None:
        return
    left_node, right_node = split_result
    train_tree(left_node, stopping_depth)
    train_tree(right_node, stopping_depth)
    return node


other_root = Adams_Node(mushDF)
tree = train_tree(other_root,2)

def print_tree(node, indent=''):
    if node == None:
        return
    if node.prediction == 0:
        poison_status = 'Edible'
    else:
        poison_status = 'Poisenous'
    print(indent + 'Depth:', node.depth, ', Feature:', node.feature, ', predictin:', poison_status)
    if node.feature == None:
        return
    print(indent + 'yes:')
    print_tree(node.left, indent + '  ')
    print(indent + 'no:')
    print_tree(node.right, indent + '  ')

print_tree(tree)

print(tree.right.left.df)

def predict(Adams_Node, x, depth):
    while Adams_Node.depth < depth:
        if x[Adams_Node.feature] == Adams_Node.left.threshold:
            Adams_Node = Adams_Node.left
        else:
            Adams_Node = Adams_Node.right
    return Adams_Node.prediction