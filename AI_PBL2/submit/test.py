import sys
import numpy as np

sys.path.append('./')

from data import *

log = True

argv = sys.argv
test_data = argv[1]

images_test = read_test_data(test_data)

predictions = 'predictions.txt'
preds = []
with open(predictions, 'rt') as f:
    while True:
        i = f.read(1)
        if i == '':
            break
        elif i.isdigit():
            preds.append(int(i))

while True:
    i = int(input('Enter index (-1 for exit): '))
    if i == -1:
        break
        
    print('Prediction: {0}'.format(preds[i]))
    show_sample(images_test[i])