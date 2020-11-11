from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#open all files and parse each line
with open('fortune/testdata.txt') as f:
    testdata = [line.rstrip() for line in f]

with open('fortune/testlabels.txt') as f:
    testlabels = [line.rstrip() for line in f]

with open('fortune/traindata.txt') as f:
    traindata = [line.rstrip() for line in f]

with open('fortune/trainlabels.txt') as f:
    trainlabels = [line.rstrip() for line in f]
    
with open('fortune/stoplist.txt') as f:
    stoplist = [line.rstrip() for line in f]    

#split the train data and test data into list of lists (bag of words)
testdata = [i.split() for i in testdata]
traindata = [i.split() for i in traindata]

#remove stop words from test and train data
for i in range(len(testdata)):
    testdata[i] = list(filter(lambda j: j not in stoplist, testdata[i]))
for i in range(len(traindata)):
    traindata[i] = list(filter(lambda j: j not in stoplist, traindata[i]))
    
#change lists of lists into list of strings
testdata = list(map(' '.join, testdata))
traindata = list(map(' '.join, traindata))

#vectorize words in string
vectorizer = CountVectorizer()
traindata = vectorizer.fit_transform(traindata)
testdata = vectorizer.transform(testdata)

#output file
textfile = ""
#loops of diff amounts of iterations 
for i in range(1,21):
    #Perceptron model
    model = Perceptron(max_iter=i,eta0=1)
    model.fit(traindata,trainlabels)
    prediction = model.predict(testdata)
    prediction_train = model.predict(traindata)
    output = "iteration-"+str(i)+" training-accuracy: "+str(round((accuracy_score(trainlabels,prediction_train)*100),2))+"%"+" testing-accuracy: "+str(round((accuracy_score(testlabels,prediction)*100),2))+"%"
    textfile = textfile + output + "\n"

#write to file    
file=open("fortune_cookie_output.txt", "w")
file.write(textfile)
file.close()        