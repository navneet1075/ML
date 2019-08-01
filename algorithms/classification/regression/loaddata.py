import pandas as pd

def load_train_data(folderName, filename):
    full_path = folderName + "/" + filename
    train_data =  pd.read_csv(full_path)
    return train_data

def load_test_data(folderName , filename):
    full_path = folderName + "/"+ filename
    test_data = pd.read_csv(full_path)
    return test_data



