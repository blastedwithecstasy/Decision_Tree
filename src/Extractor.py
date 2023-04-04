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
    # Get_dummies() one-hot encodes the relavent columns
    df = pd.get_dummies(df, columns= columnsToEncode)
    return df

mushDF = oneHotEncoder(mushDF)
print(mushDF)

def replaceWithBinary(df):
    for col in df.columns:
        uniqueVals = df[col].unique()
        # If this column is already a binary column, move on
        if set(uniqueVals) == set([0, 1]):
            continue
        # If there are more than two unique values, skip the column
        if len(uniqueVals) != 2:
            continue
        # Create a dictionary mapping that maps the "max" letter to 1 and the "min" letter to 0
        mappingDict = {max(uniqueVals): 1, min(uniqueVals): 0}
        df[col] = df[col].replace(mappingDict)
    return df

mushDF = replaceWithBinary(mushDF) 

mushDF = mushDF.astype(int)
print(mushDF)

mushDF = mushDF.drop('Fake', axis=1)
print(mushDF)


# Creating training and testing sets
np.random.seed(0)
randomized_row_nums = np.random.permutation(mushDF.shape[0])
# Training set will contain 80% of the data
split_idx = int(0.8 * len(randomized_row_nums))
traning_mush_df = mushDF.iloc[randomized_row_nums[:split_idx]]
testing_mush_df = mushDF.iloc[randomized_row_nums[split_idx:]]

print(traning_mush_df)
print(testing_mush_df)

# So that 0.5 is rounded to 1, instead of the closest even integer
def my_round(x):
    EPSILON = 1e-9  
    return round(x + EPSILON)

# This data type will represents the nodes in the tree 
class DT_Node:
    def __init__(self, dataframe,feature = None, threshold=None, parent = None, left=None, right=None, depth=0):
        # The dataframe are all the mushrooms that fall under this node
        self.df = dataframe          
        # This is the best feature of this node. The feature that we will split the node on
        self.feature = feature       
        # When a parent node is split, all mushrooms with a 1 for that feature go to the child node
        # where threshold = 1 and all the mushrooms with a 0 for that feature go to the child node
        # with threshold = 0.
        self.threshold = threshold  # Note that the root node will not have a threshold because no parent 
        # The parent of this node. Empty for root node.
        self.parent = parent         
        # The left node. Will be empty for leaf nodes
        self.left = left             
        # The right node. Will be empty for leaf nodes
        self.right = right           
        # The depth of this node in the tree.
        self.depth = depth          
        # The proportion of mushrooms in the dataframe of this node that are poisonous
        self.probability = self.df['Poisonous'].mean()
        # If a new mushroom ends up in this node, this will determine the class lable given
        self.prediction = my_round(self.probability) 

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
    # Column is None when there is no best feature to split on. 
    if column == None:
        return
    df = node.df
    left_df = df[df[column] == 1]
    right_df = df[df[column] == 0]
    # If you've already split on this feature, just return
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
    print(indent + 'Depth:', node.depth, ', predictin:', poison_status)
    if node.feature == None:
        return
    print(indent + node.feature, 'yes:')
    print_tree(node.left, indent + '  ')
    print(indent + node.feature, 'no:')
    print_tree(node.right, indent + '  ')

print_tree(tree)


# Tell me if this mushroom is poisonous or not
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



# Tell me if these mushrooms are poisonous or not
def make_predictions(tree, df):
    preds = []
    for i in range(df.shape[0]):
        x = df.iloc[i]
        pred = predict(tree, x)
        preds.append(pred)
    preds = np.array(preds)
    return preds

# Will be used to calculate evaluation measures and make ROC 
def calculate_scores(y, y_hat):
    vec = y - 2*y_hat
    tp = np.count_nonzero(vec == -1) 
    tn = np.count_nonzero(vec ==  0) 
    fp = np.count_nonzero(vec == -2) 
    fn = np.count_nonzero(vec ==  1) 
    return tp, tn, fp, fn


# Calculates accuracy, percision and recal from scores
def calculate_performance(df, y_hat):
    y_true = df['Poisonous'].to_numpy()
    tp, tn, fp, fn = calculate_scores(y_true, y_hat)
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1Score = (2*prec*rec)/(prec+rec)
    return acc, prec, rec, f1Score 

pred = make_predictions(tree, mushDF)
Accuracy, Percision, Recall,f1_score  = calculate_performance(mushDF, pred)
print('The Accuracy of the model on this dataset is : ', Accuracy)
print('The percision of the model on this dataset is :', Percision)
print('The Recal of the model on this dataset is :', Recall)
print('The F - Score of the model on this dataset is :', f1_score)


## will have to make the n-fold cross validation code and debug.


# Probability that this mushroom is poisenous. 
def give_prob(node, x):
    # If the feature that a node will be split on is 'None', then we are at a leaf node. 
    # Threfore simply read off the probability 
    if node.feature == None:
        prob = node.probability
    else:
        if x[node.feature] == 1:
            prob = give_prob(node.left, x)
        else:
            prob = give_prob(node.right, x)
    return prob 


# Probabilities of these mushrooms being poisonous
def give_probabilities(tree, df):
    probs = []
    for i in range(df.shape[0]):
        x = df.iloc[i]
        prob = give_prob(tree, x)
        probs.append(prob)
    probs = np.array(probs)
    return probs 


# Calculate TPR and FPR
def calculate_rates(df, p_hat, threshold):
    #If the value is higher than the threshold, mark it as a positive, i.e. poisonous. 
    #Else mark it as 0 i.e. eddible. 
    new_p_hat = np.where(p_hat > threshold, 1, 0)
    y_true = df['Poisonous'].astype(int).to_numpy()
    tp, tn, fp, fn = calculate_scores(y_true, new_p_hat)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


# Draw me a ROC curve
def draw_roc_curve(df, p_hat, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    tprs = []
    fprs = []
    for threshold in thresholds:
        tpr, fpr = calculate_rates(df, p_hat, threshold)
        tprs.append(tpr)
        fprs.append(fpr)

    plt.plot(fprs, tprs)
    plt.title('ROC Curve for training set')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    # For debugging
    return tprs, fprs


p_pred = give_probabilities(tree, mushDF)
print("p_preds are:")
print(p_pred)
tprs, fprs = draw_roc_curve(mushDF, p_pred)
print("TPRs:")
print(tprs)
print("FPRs:")
print(fprs)



