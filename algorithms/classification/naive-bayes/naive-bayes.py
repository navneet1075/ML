import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
import numpy as np

import os
print(os.listdir("../../dataset/naive-bayes"))

data = pd.read_csv("../../dataset/naive-bayes/tennis.csv")
outlook_count = data.groupby(['outlook', 'play']).size()
outlook_total = data.groupby(['outlook']).size()
temp_count = data.groupby(['temp', 'play']).size()
temp_total = data.groupby(['temp']).size()
humidity_count = data.groupby(['humidity', 'play']).size()
humidity_total = data.groupby(['outlook']).size()
windy_count = data.groupby(['windy', 'play']).size()
windy_total = data.groupby(['windy']).size()

print(outlook_count)
print(windy_total)
print(outlook_total)
print(temp_count)
print(temp_total)
print(humidity_count)
print(humidity_total)
print(windy_count)
print(windy_total)


p_over_yes = outlook_count['overcast','yes']
p_over_no = 0
p_rainy_yes = outlook_count['rainy','yes']
p_rainy_no = outlook_count['rainy','no']
p_rainy_yes = outlook_count['sunny', 'yes']

X_train = pd.get_dummies(data[['outlook', 'temp', 'humidity', 'windy']])
y_train = pd.DataFrame(data['play'])


model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train, y_train)

#Predict Output 
predicted= model.predict([[False,1,0,0,0,1,0,1,0]])
print (predicted)

