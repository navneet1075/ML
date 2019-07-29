from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_evaluate(final_train,Selected_features,logreg):
    X = final_train[Selected_features]
    y = final_train['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# check classification scores of logistic regression

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Train/Test split results:')
    print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))


def predict_actual(final_test,Selected_features,test_df,logreg):

    final_test['Survived'] = logreg.predict(final_test[Selected_features])
    final_test['PassengerId'] = test_df['PassengerId']
    submission = final_test[['PassengerId', 'Survived']]
    submission.to_csv("submission.csv", index=False)
    print(submission.tail())