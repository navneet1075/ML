import pandas as pd

data_base_path = "../../dataset/"

def get_data_base_path():
    return data_base_path


def load_train_data(folderName, filename):
    full_path = get_data_base_path()+ folderName + "/" + filename
    print(full_path)
    train_data =  pd.read_csv(full_path)
    return train_data

def load_test_data(folderName , filename):
    full_path = get_data_base_path() + folderName + "/"+ filename
    print(full_path)
    test_data = pd.read_csv(full_path)
    return test_data

def get_train_data():
    return load_train_data("titanic","train.csv")

def get_test_data():
    return load_test_data("titanic","test.csv")

