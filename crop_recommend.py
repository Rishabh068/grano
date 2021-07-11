import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
from sklearn import tree
df = pd.read_csv('crop_recommendation.csv')
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
print("RF's Accuracy is: ", x)




#filename= 'RandomForest.pkl'

#pickle.dump(RF,open(filename,'wb'))
#RF_pkl_filename = 'RandomForest.pkl'
#RF_Model_pkl = open(RF_pkl_filename, 'wb')
#pickle.dump(RF, RF_Model_pkl)
#RF_Model_pkl.close()
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)