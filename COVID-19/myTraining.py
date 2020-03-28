import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]



if __name__ == "__main__" :
   df = pd.read_csv('data.csv.csv')
   train, test = data_split(df,0.2)
   X_train = train[['BodyTemp','BodyPain','Age','runnyNose','DiffBreath']].to_numpy()
   X_test = test[['BodyTemp','BodyPain','Age','runnyNose','DiffBreath']].to_numpy()

   Y_train = train[['InfectionProb']].to_numpy().reshape(2037,)
   Y_test = test[['InfectionProb']].to_numpy().reshape(509 ,)
  
   clf = LogisticRegression()
   clf.fit(X_train, Y_train)
    
   file = open('model.pkl','wb')


   pickle.dump(clf, file)
   file.close()
   
   