import math

class Node:
    def _init_(self, dataframe,feature = None, threshold=None, parent = None, left=None, right=None, depth=0):
        self.df = dataframe          # the data points in this Node 
        self.feature = feature       # the feature this Node was slit from its parent Node on
        self.threshold = threshold   # the threshold to split on (1 or 0)
        self.parent = parent         # the parent of this Node. None for the root.
        self.left = left             # the left child Node
        self.right = right           # the right child Node
        self.depth = depth           # the distance of the Node form the root
        self.prediction = round(dataframe['Poisonous'].mean()) # the prediction if a new data point ends in this Node
                                     

def entropy_calculator(Node):
    df = Node.df
    n_positive = (df['Poisonous'] == 1).sum()
    n_total = df.shape[0]
    p_positive = n_positive/n_total
    p_negative = 1-p_positive 
    if (p_positive == 0 or p_positive == 1):
        return 0
    entropy = -p_positive*math.log2(p_positive) - p_negative*math.log2(p_negative)
    return entropy

def split(Node, column_num):
    if(Node == None or column_num == None):
       return
    df = Node.df
    left_df = df[df[column_num] == 1]
    right_df = df[df[column_num] == 0]
    left_node = Node(df = left_df, feature = column_num, threshold = 1, parent = Node, depth = Node.depth+1)
    right_node = Node(df = right_df, feature = column_num, threshold = 0, parent = Node, depth = Node.depth+1)
    return left_node, right_node


def fetch_best_feature(Node):
    if(Node == None):
      return
    df = Node.df
    n_positive = (df['Poisonous'] == 1).sum()
    n_total = df.shape[0]
    p_positive = n_positive/n_total
    p_negative = 1-p_positive 
    curent_entropy = entropy_calculator(Node)
    max_information_gain = 0
    for column in Node.df.columns:
        left_node, right_node = split(Node, column)
        # If you've already split on that feature skip
        if left_node.df.empty or right_node.df.empty:
            continue
        information_gain = curent_entropy - p_positive*entropy_calculator(left_node) -p_negative*entropy_calculator(right_node)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = column
    return best_feature


def train_tree(Node, stopping_depth = 1):
    if Node == None or Node.depth >= stopping_depth:
      return
    feature = fetch_best_feature(Node)
    left_node, right_node = split(Node, feature)
    train_tree(left_node, stopping_depth)
    train_tree(right_node, stopping_depth)
    return Node 

def predict(Node, x, depth):
    while Node.depth < depth:
        if x[Node.feature] == Node.threshold:
            Node = Node.left
        else:
            Node = Node.right
    return Node.value


def print_tree(node,depth, indent=''):
    if Node.depth == depth:
        print(indent + 'Leaf:', node.value)
        return
    print(indent + 'Feature:', node.feature, 'Threshold:', node.threshold)
    print(indent + 'Left:')
    print_tree(node.left,depth, indent + '  ')
    print(indent + 'Right:')
    print_tree(node.right,depth, indent + '  ')

# Depth 0, Root: Split on feature: 2
# - Depth 1, Left: Split on feature: 0
#   -- Left leaf node with indices [0, 1, 4, 7]
#   -- Right leaf node with indices [5]
# - Depth 1, Right: Split on feature: 1
#   -- Left leaf node with indices [8]
#   -- Right leaf node with indices [2, 3, 6, 9]


    

