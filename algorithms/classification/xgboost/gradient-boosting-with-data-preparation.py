from numpy import loadtxt
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import pandas



def xgboost_with_data_preparation_train_eval():
    data = pandas.read_csv('../dataset/iris/iris.data', header=None)
    dataset = data.values
    X = dataset[:, 0:4] #features
    Y = dataset[:, 4] #output
    # the Y values are strings and xgboost requires the output to be in numeric format
    # so labelling the output in integers using the labelencoder
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(Y)
    label_encoded_y = label_encoder.transform(Y)
    print(label_encoded_y)

    train_test_split_ratio = 0.3
    seed = 7
    X_train, X_test, y_train, y_test = train_test_split(X,label_encoded_y,test_size=train_test_split_ratio,random_state=seed)

    # prepare the model
    model = XGBClassifier() # using default values in the beginning , parameter tuning will be done later
    model.fit(X_train,y_train)
    print(model)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test,predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))



if __name__ == '__main__':
    xgboost_with_data_preparation_train_eval()