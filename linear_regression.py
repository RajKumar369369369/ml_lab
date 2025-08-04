import numpy as np
import pandas as pd
df=pd.read_csv('advertising.csv')
print(df)
X= df[['TV', 'Radio', 'Newspaper']].values
y_actual =df['Sales'].values
print(y_actual.shape)
X = np.hstack((np.ones((X.shape[0], 1)), X))
#print(X)
weights =np.random.rand(X.shape[1]).reshape(-1,1)
#print(weights.shape)
#print(weights)

def y_predict(X, weights):
    return X @ weights
y_predicted = y_predict(X, weights)
print(y_predicted.shape)

#print(y_predicted)
def mse_loss(y_actual, y_predicted):
    return np.mean((y_actual.reshape(-1,1) - y_predicted) ** 2)

print(mse_loss(y_actual, y_predicted))

def weight_gradients(x, y_actual, weights):
    y_predicted = y_predict(x, weights).reshape(1, -1)
    return -2/(len(y_actual))*(y_actual-y_predicted)@x


#hel=np.random.choice(X.shape[0], 5, replace=False)
hel=[1,2,3,4,5]
"""print(hel)
print(y_actual[hel].shape)
print(X[hel])"""
w=weight_gradients(X[hel], y_actual[hel].reshape(1,-1), weights)
print(w)
"""n= np.array([[1,2,3,3],[2,3,3,4],[3,4,5,6],[4,5,6,7],[5,6,7,8]])
print(n[[2,4]])"""

def mini_batch_gradient_descent(X, y_actual, weights, learning_rate=0.01):
        indexes= np.arange(len(y_actual))
        np.random.shuffle(indexes)
        batches= indexes.reshape(-1, 5)
        for batch in batches:
            X_shuffled = X[batch]
            y_actual_shuffled = y_actual[batch]
            gradients = weight_gradients(X_shuffled, y_actual_shuffled, weights)
            weights -= learning_rate * gradients.T

for i in range(100):
 print(f'epoch {i+1} loss:{mse_loss(y_actual, y_predict(X, weights))}')
 mini_batch_gradient_descent(X, y_actual, weights, learning_rate=0.001)

 def online_gradient_descent(X, y_actual, weights, learning_rate=0.01):
     for i in range(X.shape[0]):
         weights -= learning_rate*2*(y_actual[i]-y_predict(X[i].reshape(1, -1), weights))*X[i].reshape(-1, 1)



