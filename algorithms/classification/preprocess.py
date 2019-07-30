import loaddata
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

def preprocess_train_data():
    train_data = loaddata.load_train_data("./data/titanic-dataset","train.csv")
    print(' the number of samples in the training data is {}', train_data.shape[0])

    # applying imputation to age by using median
    train_df = train_data.copy()
    train_df['Age'].fillna(train_data["Age"].median(skipna=True),inplace=True)

    # if embarked is missing for a given row then apply the port of most embarkation in that row
    train_df['Embarked'].fillna(train_data['Embarked'].value_counts().idxmax(),inplace=True)

    # dropping cabin column as most of the values are null
    train_df.drop('Cabin', axis=1, inplace=True)

    # check missing values in the adjusted data
    print('adjusted misssing value', train_df.isna().sum())

    # create additional variables
    train_df['TravelAlone'] = np.where(train_df['SibSp']+ train_df['Parch']>0 , 0,1)
    train_df.drop('SibSp', axis=1, inplace=True)
    train_df.drop('Parch', axis=1, inplace=True)

    # create categorical variables
    training = pd.get_dummies(train_df,columns=['Pclass','Embarked','Sex'])
    training.drop("Sex_female", axis=1, inplace=True)
    training.drop("PassengerId", axis=1, inplace=True)
    training.drop("Name", axis=1, inplace=True)
    training.drop("Ticket", axis=1, inplace=True)
    final_training = training
    print(final_training.head())
    final_training['IsMinor'] = np.where(final_training['Age'] <= 16, 1, 0)
    feature_selection(final_training)
    return  final_training


def preprocess_test_data():
    test_data = loaddata.load_test_data("./data/titanic-dataset", "test.csv")

    print(' the number of samples in the training data is {}', test_data.shape[0])
    test_df = test_data.copy()
    test_df['Age'].fillna(test_data["Age"].median(skipna=True), inplace=True)
    test_df["Fare"].fillna(test_data["Fare"].median(skipna=True), inplace=True)
    test_df.drop('Cabin', axis=1, inplace=True)

    test_df['TravelAlone'] = np.where((test_data["SibSp"] + test_data["Parch"]) > 0, 0, 1)

    test_df.drop('SibSp', axis=1, inplace=True)
    test_df.drop('Parch', axis=1, inplace=True)

    testing = pd.get_dummies(test_df, columns=["Pclass", "Embarked", "Sex"])
    testing.drop('Sex_female', axis=1, inplace=True)
    testing.drop('PassengerId', axis=1, inplace=True)
    testing.drop('Name', axis=1, inplace=True)
    testing.drop('Ticket', axis=1, inplace=True)

    final_test = testing

    final_test['IsMinor'] = np.where(final_test['Age'] <= 16, 1, 0)
    return final_test

def feature_selection(final_train_test):


    cols = ["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2", "Embarked_C", "Embarked_S", "Sex_male", "IsMinor"]
    X = final_train_test[cols]
    y = final_train_test['Survived']
    # Build a logreg and compute the feature importances
    model = LogisticRegression()
    # create the RFE model and select 8 attributes
    rfe = RFE(model, 8)
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    print('Selected features: %s' % list(X.columns[rfe.support_]))



def data_analysis():
    train_data = loaddata.load_train_data("./data/titanic-dataset","train.csv")
    missing_Values = train_data.isna().sum()
    print(missing_Values)

    # missing values for a particular row
    missing_values_age = train_data['Age'].isna().sum()
    missing_values_cabin = train_data['Cabin'].isna().sum()

    print('missing values in the age ', missing_values_age)
    print ('percent of missing age in the train data ', missing_values_age/train_data.shape[0]*100)
    print('percent of missing cabin values in the train data' , missing_values_cabin/train_data.shape[0]*100)

    # embarked missing values
    missing_values_embarked = train_data['Embarked'].isna().sum()
    print('missing embarked ', missing_values_embarked)
    print('percentage of embarked values missing in the training data is ',missing_values_embarked/train_data.shape[0]*10)

    # boarded passengers grouped by the port of embarkation

    print('Boarded passengers grouped by the port of embarkation' , train_data['Embarked'].value_counts())
    # the most common port of embarkation
    print(' the most common port of embarkation is ', train_data['Embarked'].value_counts().idxmax())