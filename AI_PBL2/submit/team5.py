import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.append('./')

from data import *
from svm import MySVM
from features import *

log = True

argv = sys.argv
if len(argv) != 4:
    print('USAGE: python {0} {1} {2} {3}'.format(argv[0], 'train-img', 'train-lbl', 'test-img'))
    exit(0)

training_data_img = argv[1]
training_data_lbl = argv[2]
test_data = argv[3]

### Data read ###

labels_train, images_train = read(training_data_img, training_data_lbl)
images_test = read_test_data(test_data)

# if log:
#     print(images_train.shape)
#     print(labels_train.shape)
#     print(images_test.shape)

# show_sample(images_test[22318])

#################

### Feature engineering ###

X_train, X_test = make_poly_features(images_train, images_test)
y_train = labels_train

# if log:
#     print(X_train.shape)
#     print(X_test.shape)

###########################

### Training ###

# svm = MySVM(C=13000, max_iter=20000, batch_size=128)
# svm.fit(X_train, y_train, log=True)
svm = MySVM(C=8000, batch_size=128, max_iter=160, eta=0.1)
svm.fit(X_train, y_train, log=True)
# svm = MySVM(C=100, max_iter=1000, batch_size=256)
# svm.fit(X_train, y_train, log=True)

if log:
    print(svm.score(X_train, y_train))

print(svm.params)

################

### Testing ###

pred = svm.predict(X_test)

with open('predictions.txt', 'w') as f:
    for i in range(len(pred)):
        f.write("{0}\n".format(pred[i]))

###############