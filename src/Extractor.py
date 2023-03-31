import math
from matplotlib import pyplot as plt
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


class DT_Node:
    def __init__(self, dataframe,feature = None, threshold=None, parent = None, left=None, right=None, depth=0):
        self.df = dataframe          
        self.feature = feature       
        self.threshold = threshold   
        self.parent = parent         
        self.left = left             
        self.right = right           
        self.depth = depth          
        self.prediction = my_round(self.df['Poisonous'].mean()) 

root = DT_Node(mushDF)
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
    left_node = DT_Node(dataframe = left_df, threshold = 1, parent = node, depth = node.depth+1)
    right_node = DT_Node(dataframe = right_df, threshold = 0, parent = node, depth = node.depth+1)
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


other_root = DT_Node(mushDF)
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


def predict(node, x):
    # If the feature that a node will be split on is 'None', then we are at a leaf node. 
    # Threfore simply read off the prediction
    if node.feature == None:
        pred = node.prediction
    else:
        if x[node.feature] == 1:
            pred = predict(node.left, x)
        else:
            pred = predict(node.right, x)
    return pred


print('X:')
x = mushDF.iloc[4]
print(x)
x_pred = predict(tree, x)
print('Poisenous?')
print(x_pred)


def make_predictions(tree, X):
    preds = []
    for i in range(X.shape[0]):
        x = X.iloc[i]
        pred = predict(tree, x)
        preds.append(pred)
    preds = np.array(preds)
    return preds


def calculate_performance(df, y_pred):
    y_true = df['Poisonous'].to_numpy()
    vec = y_true - 2*y_pred
    tp = np.count_nonzero(vec == -1) 
    tn = np.count_nonzero(vec ==  0) 
    fp = np.count_nonzero(vec == -2) 
    fn = np.count_nonzero(vec ==  1) 
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    return acc, prec, rec 

pred = make_predictions(tree, mushDF)
Accuracy, Percision, Recall  = calculate_performance(mushDF, pred)
print('The Accuracy of the model on this dataset is : ', Accuracy)
print('The percision of the model on this dataset is :', Percision)
print('The Recal of the model on this dataset is :', Recall)


## will have to make the n-fold cross validation code and debug.
## will have to make the roc_curve, debug and plot.

def roc_curve(y_true, y_pred, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    tprs = []
    fprs = []

    for threshold in thresholds:
        tpr, fpr = calculate_rates(y_true, y_pred, threshold)
        tprs.append(tpr)
        fprs.append(fpr)
    
    plt.plot(fprs, tprs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


