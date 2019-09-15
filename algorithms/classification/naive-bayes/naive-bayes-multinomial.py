from sklearn import datasets
wine = datasets.load_wine()

print ("Features: ", wine.feature_names)

print("Labels", wine.target_names)