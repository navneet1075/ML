from sklearn.linear_model import LogisticRegression

from classification.regression.preprocess import preprocess_train_data, data_analysis,preprocess_test_data
from classification.regression.train_and_evaluate import  train_and_evaluate,predict_actual
from classification.regression.loaddata import get_test_data
if __name__ == '__main__':
    test_data = get_test_data()
    data_analysis()
    final_train = preprocess_train_data()
    final_test= preprocess_test_data()

    Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C',
                         'Embarked_S', 'Sex_male', 'IsMinor']
    logreg = LogisticRegression()
    train_and_evaluate(final_train,Selected_features,logreg)
    predict_actual(final_test,Selected_features,test_data,logreg)