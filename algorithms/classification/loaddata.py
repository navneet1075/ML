import pandas as pd

def load_train_data(folderName, filename):
    full_path = folderName + "/" + filename
    train_data =  pd.read_csv(full_path)
    return train_data

def load_test_data(folderName , filename):
    full_path = folderName + "/"+ filename
    test_data = pd.read_csv(full_path)
    return test_data


#
# if __name__ == '__main__':
#     train_data = load_train_data("./data/titanic-dataset", 'train.csv')
#     test_data = load_test_data("./data/titanic-dataset", 'test.csv')
#     print(train_data.head())
#     print(test_data.head())


