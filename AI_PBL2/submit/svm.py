import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MySVM(BaseEstimator, ClassifierMixin):
    """
    1 vs 1 SVM (binary classification)
    """
    def __init__(self, C=0.1, eta=0.001, batch_size=1, max_iter=25, epsilon=1e-8):
        """
        constructor of MyBinarySVM class.
        """
        
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.eta = eta
        self.C = C
        self.epsilon = epsilon
        self.num_classes = 0
#         self.beta1 = 0.9
#         self.beta2 = 0.99
    
    def fit(self, X, y=None, params=None):
        """
        fit method for training svm
        
        Arguments:
        --------------------------
        X: image data. (60000, 784)
        y: label data. (60000, 1)
        
        Returns:
        --------------------------
        Z: class score
        """
        
        m = np.shape(X)[0]
        n = np.shape(X)[1]
        self.num_classes = len(np.unique(y))
        
        y_encoded = self.encode_y(y)
        
        # create weights.
        if params is None:
            self.params = {
                'W': np.random.randn(n, self.num_classes),
                'b': np.random.randn(1, self.num_classes)
            }
#             self.M = {
#                 'W': np.zeros((n, self.num_classes)),
#                 'b': np.zeros((1, self.num_classes))
#             }
#             self.V = {
#                 'W': np.zeros((n, self.num_classes)),
#                 'b': np.zeros((1, self.num_classes))
#             }

        cnt = 1
        
        # main loop: how much iterate on entire dataset.
        for epoch in range(self.max_iter):
            # before dive into SGD, shuffle dataset
            X_shuffled, y_shuffled = self.shuffle(X, y_encoded)
            
            # cost variable for printing/logging
            avg_loss = 0
            
            # batch_count = dataset_size / batch_size
            batch_count = int(np.ceil(np.shape(X)[0] / self.batch_size))
            
            # mini-batch loop
            for t in range(batch_count):
                # draw the {batch_size} number of samples from X and y
                X_batch, y_batch, bs = self.next_batch(X_shuffled, y_shuffled, t)
                
                # just in case, reshape batch of X and y into proper shape.
                X_batch = np.reshape(X_batch, (bs, n))
                y_batch = np.reshape(y_batch, (bs, self.num_classes))
                
                # prediction phase
                Z = self.forward_prop(X_batch)
                Z = np.reshape(Z, (bs, self.num_classes))
                
                # compute cost phase
                loss = self.compute_cost(y_batch, Z)
                
                # update weights phase
                self.backward_prop(X_batch, y_batch, Z, bs, cnt)
                
                # accumulate loss
                avg_loss += loss
                cnt += 1
        
            # logging
            avg_loss /= batch_count
            if epoch % (self.max_iter / 10) == 0:
                print('Cost at epoch {0}: {1}'.format(epoch, avg_loss))

        return self
    
    def encode_y(self, y):
        y_encoded = np.ones((np.shape(y)[0], self.num_classes))
        
        for i in range(self.num_classes):
            y_encoded[:, i][y != i] = -1
            
        return y_encoded
    
    def shuffle(self, X, y):
        """
        Random selection is required for SGD.
        But, my approach is to shuffle entire data before every iteration.
        This has same effect as random selection.
        
        Arguments:
        ---------------------------
        X: images (BATCH_SIZE, 784)
        y: labels (BATCH_SIZE, 1)
        
        Returns:
        ---------------------------
        shuffled data
        """
        
        # the number of dataset samples
        m = np.shape(X)[0]
        
        # variable for shuffle
        r = np.arange(0, m)
        
        np.random.shuffle(r)
        
        return X[r], y[r]
    
    def next_batch(self, X, y, t):
        """
        Get next batch.
        If it is SGD, next_batch function just pick one sample from dataset.
        
        Arguments:
        ---------------------------------
        X: images (60000, 784)
        y: labels (60000, 1)
        
        Returns:
        ---------------------------------
        X_batch: small subset of X (BATCH_SIZE, 784)
        y_batch: small subset of y (BATCH_SIZE, 1)
        """
        
        # the number of dataset samples
        m = np.shape(X)[0]
        
        # draw the {batch_size} number of samples from X and y
        X_batch = X[t * self.batch_size : min(m, (t + 1) * self.batch_size)]
        y_batch = y[t * self.batch_size : min(m, (t + 1) * self.batch_size)]
        bs = min(m, (t + 1) * self.batch_size) - t * self.batch_size
        
        return X_batch, y_batch, bs
    
    def forward_prop(self, X):
        """
        Process of inference (prediction).
        
        Arguments:
        -----------------------
        X: images e.g (BATCH_SIZE, 784)
        params: weights dictionary(map in other programming language)
        
        Returns:
        -----------------------
        A: 
        """
        
        # prediction
        Z = np.matmul(X, self.params['W']) + self.params['b']
        
        return Z
    
#     def sigmoid(self, Z):
#         """
#         sigmoid activation for binary classification
        
#         Arguments:
#         ----------------------
#         Z: class score (W.T * X)
        
#         Returns:
#         ----------------------
#         sigmoid activation
#         """
        
#         return 1 / (1 + np.exp(-Z))
    
    def compute_cost(self, y, Z):
        """
        compute cost function (loss function)
        
        Arguments:
        ------------------------------
        y: true label
        Z: class score (W.T * X)
        
        Returns:
        ------------------------------
        loss: total cost (loss)
        """
        
        # compute loss function
        temp = 1 - np.multiply(y, Z)
        temp[temp < 0] = 0
        loss = np.mean(temp)
        return loss
    
    def backward_prop(self, X, y, Z, bs, cnt):
        """
        update weights
        
        Arguments:
        ----------------------------
        X: images e.g (BATCH_SIZE, 784)
        y: labels e.g (BATCH_SIZE, 1)
        Z: class score after forward propagation
        params: weights dictionary(map in other programming language)
        eta: learning rate
        
        Returns:
        ----------------------------
        params: weights dictionary
        """
        
        # number of features
        n = np.shape(X)[1]
        
        # differential vector of loss function to update weights
        dw = np.zeros(self.params['W'].shape)
        db = np.zeros(self.params['b'].shape)
        
        Z = np.reshape(Z, (bs, self.num_classes))
        temp = np.multiply(y, Z)
        temp = 1 - temp
        
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        
        y_temp = np.multiply(y, temp.reshape(bs, self.num_classes))
        
        dw = -(1 / bs) * np.matmul(X.T, y_temp) + (1 / self.C) * self.params['W']
        db = -(1 / bs) * np.sum(y_temp, axis=0)
        
#         if cnt == 1:
#             self.M['W'] = dw
#             self.M['b'] = db
#         else:
#             self.M['W'] = (self.beta1 * self.M['W'] + (1 - self.beta1) * dw)
#             self.M['b'] = (self.beta1 * self.M['b'] + (1 - self.beta1) * db)
        
#         if cnt == 1:
#             self.V['W'] = dw ** 2
#             self.V['b'] = db ** 2
#         else:
#             self.V['W'] = (self.beta2 * self.V['W'] + (1 - self.beta2) * (dw ** 2))
#             self.V['b'] = (self.beta2 * self.V['b'] + (1 - self.beta2) * (db ** 2))
    
#         self.M['W'] = (self.beta1 * self.M['W'] + (1 - self.beta1) * dw) / (1 - self.beta1 ** cnt)
#         self.M['b'] = (self.beta1 * self.M['b'] + (1 - self.beta1) * db) / (1 - self.beta1 ** cnt)
        
#         self.V['W'] = (self.beta2 * self.V['W'] + (1 - self.beta2) * np.square(dw)) / (1 - self.beta2 ** cnt)
#         self.V['b'] = (self.beta2 * self.V['b'] + (1 - self.beta2) * np.square(db)) / (1 - self.beta2 ** cnt)

        # update weights
#         self.params['W'] = self.params['W'] - np.divide(self.eta * self.M['W'], np.sqrt(self.V['W']) + self.epsilon)
#         self.params['b'] = self.params['b'] - np.divide(self.eta * self.M['b'], np.sqrt(self.V['b']) + self.epsilon)
        
        self.params['W'] = self.params['W'] - (self.eta / (1 + self.epsilon * cnt)) * dw
        self.params['b'] = self.params['b'] - (self.eta / (1 + self.epsilon * cnt)) * db

#         self.params['W'] = self.params['W'] - (self.eta / (1 + self.epsilon * cnt)) * dw
#         self.params['b'] = self.params['b'] - (self.eta / (1 + self.epsilon * cnt)) * db
        
        return self.params
    
    def predict(self, X, y=None):
        m = np.shape(X)[0]
        
        class_score = self.forward_prop(X)
        pred = np.argmax(class_score, axis=1)
        
        return pred
    
    def score(self, X, y=None):
        pred = self.predict(X)
        score = np.mean(pred == y)
        
        return score
    
    def get_parameters(self):
        return self.params