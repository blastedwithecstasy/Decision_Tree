import math

import pandas as pd

df = pd.read_csv('C:/Users/adamg/361_A1/trail.csv')
print(df)

def entropy(pt):
  if(pt == 0 or pt == 1):
    return 0
  pf = 1-pt
  return -pt*math.log2(pt)-pf*math.log2(pf)

def infgain(ethis, el, er, wl,wr):
  return ethis - wl*el - wr*er


# n = df.shape[0]
n = df.shape[0]
n1_df = sum(df['OK'])
p1_df = n1_df/n
df_entropy = entropy(p1_df) 
print('Before the split the entropy is ', round(df_entropy,4),'.')
highest_IG = 0
column_to_split_on = ''

# for column in df
for column in df.columns:
  if column == 'OK':
    continue
#  split the df into two dfs based on if you see a 1 or a 0 
  leftdf  = df[df[column] == 1]
  rightdf = df[df[column] == 0]
#  nyes = yes.shape[0], nno = no.shape[0], wl = nyes/n, wr = nno/n
  n_left = leftdf.shape[0]
  n_right = rightdf.shape[0] 
  wl = n_left/n 
  wr = n_right/n
#  count the number of 1s in that column in yesdf, p1 = #nyes/nyesdf, calculate entropy for dfyes. Do the same for df no.
  n_okay_left = sum(leftdf['OK'])
  p1_left = n_okay_left/n_left
  entropy_left = entropy(p1_left)

  n_okay_right = sum(rightdf['OK'])
  p1_right = n_okay_right/n_right
  entropy_right = entropy(p1_right)
#   calculate the informaiton gain
  IG_for_this_feature = infgain(df_entropy, entropy_left, entropy_right, wl, wr)
  if IG_for_this_feature > highest_IG:
    highest_IG = IG_for_this_feature
    column_to_split_on = column

# print('spliting on', column)
  print('If we decide to split on ',column)
  print('For the trails with', column ,', proportion(OK)= ', round(p1_left,4) ,', Entropy(p_okay) = ', round(entropy_left, 4))
  print('And for the trails that are not ',column, ', proportion(OK) = ', round(p1_right,4), ', Entropy(p_okay)', round(entropy_right,4))
# print(The information gain after spliting on this feature is, )
  print('Therefore the information gain after splitting on ', column, 'will be IG = ', round(IG_for_this_feature,4))

print('And so as we can see spliting on ', column_to_split_on, 'will yeald the hightest information gain of', round(highest_IG,4))


#Split based on the highest information gain. 
#Choose the column with the highest information gain and split the dataset into two. Representing the diffrent nodes. 
#repeat the same procedure for the other datasets untill you have a tree. 




