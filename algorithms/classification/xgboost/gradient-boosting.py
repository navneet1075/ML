from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging


def xgboost_load_train_eval():
    dataset = loadtxt('../dataset/pima-indians-diabetes/pima-indians-diabetes.data.csv', delimiter=',')
    X = dataset[:, 0:8]  # features
    Y = dataset[:, 8]  # output

    # split using train test split
    test_size = 0.33
    seed = 20
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # create the model

    # model = XGBClassifier() # leaving sensible defaults // accuracy = 0.7559055118110236

    # parameter tuning and performance

    # model = XGBClassifier(max_depth=1) # reducing the depth of the gbm tree 0.7716535433070866

    model = XGBClassifier(n_estimators=100, max_depth=1)

    model.fit(X_train, y_train)  # training the module

    print(model)

    # making prediction on test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    print(y_pred)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)


def main():
    xgboost_load_train_eval()


if __name__ == '__main__':
    main()
