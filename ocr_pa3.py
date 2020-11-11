from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd

#open all files as dataframe
testdata = (pd.read_csv('OCR-data/ocr_test.txt',sep='\t',names=[0,1,2,3])).dropna()
traindata = (pd.read_csv('OCR-data/ocr_train.txt',sep='\t',names=[0,1,2,3])).dropna()

#extract needed columns
train_data = (traindata[1].str[3:]).tolist()
train_label = (traindata[2]).tolist()
test_data = (testdata[1].str[3:]).tolist()
test_label = (testdata[2]).tolist()

#convert into list of list ints
for i in range(len(train_data)):
    train_data[i] =  list(map(lambda j:int(j), train_data[i])) 
for i in range(len(test_data)):
    test_data[i] =  list(map(lambda j:int(j), test_data[i])) 

#output file
textfile = ""

#loops of diff amounts of iterations 
for i in range(1,21):
    #Perceptron model
    model = Perceptron(max_iter=i,eta0=1)
    model.fit(train_data,train_label)
    prediction = model.predict(test_data)
    prediction_train = model.predict(train_data)
    output = "iteration-"+str(i)+" training-accuracy: "+str(round((accuracy_score(train_label,prediction_train)*100),2))+"%"+" testing-accuracy: "+str(round((accuracy_score(test_label,prediction)*100),2))+"%"
    textfile = textfile + output + "\n"

#write to file    
file=open("ocr_output.txt", "w")
file.write(textfile)
file.close()        