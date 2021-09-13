# Import wine dataset
import pandas as pd
wines = pd.read_csv('winequality-red.csv')
wines.columns = wines.columns.str.replace(" ", "_")

# Split dataset into features and target
X = wines.loc[:, ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'alcohol']]
y = wines.loc[:, 'quality']

# Scale the variables to be within the range of -1 to 1.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X)
X = scaler.transform(X)

# Train a Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Print the model score
model.score(X, y)

# Export the model using pickle
import pickle
file_name = "model.pkl"
open_file = open(file_name, "wb")
pickle.dump([scaler, model], open_file)
open_file.close()