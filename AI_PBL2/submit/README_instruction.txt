##### Training & testing #####

python ./Team5.py {training_image_file} {training_label_file} {test_image_file}

ex: 
python Team5.py data/train-images.idx3-ubyte data/train-labels.idx1-ubyte data/mnist_new_testall-patterns-idx3-ubyte

###############################



##### Comparing with images #####

python test.py {test_image_file}

ex:
python test.py data/mnist_new_testall-patterns-idx3-ubyte

##################################