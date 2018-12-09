# Import Important Libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# SVM Model
class SVM:
    
    # Initialize the corresponding Properties of class
    def __init__(self,X,Y,c = 1.0):
        
        self.X = X
        self.Y = Y
        self.b = 0
        self.c = c
        self.W = np.random.random((X.shape[1]))
        
    # To calculate the Hypothesis or hx
    def hypothesis(self):
        return np.dot(self.W,self.X.T) + self.b
    
    # To calculate the loss calculated
    def hingeloss(self):
        
        loss = 0
        loss += 0.5*np.dot(self.W,self.W.T)

        hx = self.hypothesis()
        ti = np.multiply(self.Y,hx)
        ti = 1 - ti
        ti = ti[ti > 0]
        loss += np.sum(ti)

        return loss
    
    # To train the model 
    def fit(self,lr = 0.0001,epochs = 700,batch_size=100,threshold=0.01):
    
        features = self.X.shape[1]
        samples = self.X.shape[0]


        losses = []
        prev_loss = 0

        epoch = 0
        
        diff = 0
        while epoch < 2 or diff > threshold:

            curr_loss = self.hingeloss()
            losses.append(curr_loss)
            
            if epoch >= 1:
                diff = abs(curr_loss-prev_loss)
            

            for batch in range(0,samples,batch_size):

                gradw = np.zeros((features,))
                gradb = 0

                Xi = self.X[batch:batch+batch_size]
                Yi = self.Y[batch:batch+batch_size]

                hx = np.dot(self.W,Xi.T) + self.b
                ti = np.multiply(Yi,hx)

                for jx in range(features):

                    for ix in range(ti.shape[0]):

                        if ti[ix] >= 1:

                            gradw[jx] += 0
                            gradb += 0

                        else:

                            gradw[jx] += self.c*Xi[ix][jx]*Yi[ix]
                            gradb += self.c*Yi[ix]

                self.W = self.W - lr*self.W + lr*gradw
                self.b = self.b + lr*gradb
            prev_loss = curr_loss
            epoch += 1
    
        return losses
    
    # To calculate the accuracy of the model over the testing data
    def accuracy(self,x_test,y_test):
    
        hx = np.dot(self.W,x_test.T) + self.b
        hx[hx >= 0] = 1
        hx[hx < 0] = -1

        return float((hx[hx == y_test].shape[0])/y_test.shape[0])   
# Reading data Set
df = pd.read_csv("mushrooms.csv")

""" Uncomment to see the dataSet
	df.head() """

""" Label Encoder to Encode the characters given
	in the dataSet into Values """

Le = LabelEncoder()
encoded_data = df.apply(Le.fit_transform)

""" Uncomment to see the Encoded Data
	encoded_data.head() """

# Convert the data Frame into numpy and Shuffle the data
data = encoded_data.values
data = np.array(data)
np.random.shuffle(data)

""" Uncomment to see the updated data
	print(data) """

# Checking only for 200 samples(You can according to your wish)
X = data[:200,1:]
Y = data[:200,0]

# Converting the labels having 0 into -1
Y[Y == 0] = -1 # BroadCasting

# Dividing the data into train and test data(80% training_data and 20% testin_data)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

""" Uncomment to see dimensions of the training and testing data 
	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape) """

# Instantiating the classifier for the model
svm = SVM(x_train,y_train,c = 10.0)

# Training of the data
losses = svm.fit()

# Accuracy of the Classifier
print("Implemented Model's Score = ",svm.accuracy(x_test,y_test))

# Using the Sklearn Classifier to test the accuracy over the given data

# Classifier for sklearn's SVM
clf = SVC()

# Training of the classifier
clf.fit(x_train,y_train)

# Accuracy of the Sklearn's Classifier
print("Sklearn's model Score =",clf.score(x_test,y_test))

# Plotting the Loss Curve
plt.plot(losses)
plt.show()