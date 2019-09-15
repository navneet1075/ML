from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

def merge(list1, list2):    
    merged_list = [(p1, p2) for idx1, p1 in enumerate(list1)  
    for idx2, p2 in enumerate(list2) if idx1 == idx2] 
    return merged_list 
      

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

le = preprocessing.LabelEncoder()

weather_encoded=le.fit_transform(weather)
temp_encoded = le.fit_transform(temp)
play_encoded = le.fit_transform(play)

print ('weather encoded', weather_encoded)
print('temp encoded ', temp_encoded)
print('play encoded', play_encoded)

features=merge(weather_encoded,temp_encoded)

for values in features:
    print(values)

model = GaussianNB()

# Train the model using the training sets
model.fit(features,play_encoded)

#Predict Output
predicted= model.predict([[0,2]])
print ("Predicted Value:", predicted)

