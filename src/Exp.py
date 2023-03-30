
# def n_fold_cross_validation(data, n):
#     # Split data into n subsets
#     subsets = np.array_split(data, n)
#     scores = []
#     for i in range(n):
#         # Select the i-th subset as test data
#         test_data = subsets[i]
#         # Concatenate the other subsets as training data
#         train_data = pd.concat([subset for j, subset in enumerate(subsets) if j != i])
#         # Train the decision tree on the training data
#         tree = build_tree(train_data)
#         # Evaluate the tree on the test data
#         score = evaluate(tree, test_data)
#         scores.append(score)
#     return np.mean(scores)

# def evaluate(tree, data):
#     # Evaluate the decision tree on the given data
#     correct = 0
#     for _, row in data.iterrows():
#         if predict(tree, row) == row['y']:
#             correct += 1
#     return correct / len(data)

