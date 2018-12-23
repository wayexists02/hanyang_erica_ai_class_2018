import numpy as np

def zero_padding(images, n=1):
    """
    zero-padding to image.
    add additional edge which has value of 0
    
    Arguments:
    ---------------------
    - images: training dataset images. maybe (60000, 28, 28)
    - n: how many padding do you want? in other word, how many edge do you want to insert?
    
    Returns:
    ---------------------
    - images_padded: padded images. (60000, 30, 30) or other shape.
    """
    
    # number of training examples. 60000. if you use test data, 10000.
    m = images.shape[0]
    
    # define larger size of window than size of images. maybe (60000, 30, 30), (60000, 32, 32)
    images_padded = np.zeros((m, images.shape[1] + 2 * n, images.shape[2] + 2 * n))
    
    # insert image in the middle of this window.
    images_padded[:, n : images_padded.shape[1] - n, n : images_padded.shape[2] - n] = images
    
    return images_padded

def x_gradient_slice(images_slice):
    """
    find gradient(in korean, 기울기 또는 미분값) for part of images.
    
    Arguments:
    ----------------------
    - images_slice: small window extracted from images. (60000, 7, 7)
    
    Returns:
    ----------------------
    - grad: x-axis-oriented gradient (in korean, x 축 방향 기울기)
    """
    
    x_gradient_filter = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1],
    ])
    
    # reshape for broadcasting.
    x_gradient_filter = x_gradient_filter.reshape(1, 3, 3)
    
    # element-wise compute. compute gradient
    temp = np.multiply(images_slice, x_gradient_filter)
    grad = np.sum(temp, axis=(1, 2))
    
    return grad

def x_gradient(images):
    """
    find gradient(in korean, 기울기 또는 미분값) for whole images.
    
    Arguments:
    ----------------------
    - images: images. (60000, 28, 28)
    
    Returns:
    ----------------------
    - grad: x-axis-oriented gradient (in korean, x 축 방향 기울기)
    """
    
    # some useful variables.
    m = images.shape[0]
    width = images.shape[1]
    height = images.shape[2]
    
    # define placeholder to store gradients.
    x_grads = np.zeros((m, width - 2, height - 2))
    
    # slice image into small size window, then compute gradient.
    for w in range(1, width - 1):
        for h in range(1, height - 1):
            images_slice = images[:, w - 1 : w + 2, h - 1 : h + 2]
            x_grads[:, w - 1, h - 1] = x_gradient_slice(images_slice)
            
    return x_grads

def y_gradient_slice(images_slice):
    """
    find gradient(in korean, 기울기 또는 미분값) for part of images.
    
    Arguments:
    ----------------------
    - images_slice: small window extracted from images. (60000, 7, 7)
    
    Returns:
    ----------------------
    - grad: y-axis-oriented gradient (in korean, y 축 방향 기울기)
    """
    
    y_gradient_filter = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ])
    
    # reshape for broadcasting.
    y_gradient_filter = y_gradient_filter.reshape(1, 3, 3)
    
    # element-wise compute. compute gradient
    temp = np.multiply(images_slice, y_gradient_filter)
    grad = np.sum(temp, axis=(1, 2))
    
    return grad

def y_gradient(images):
    """
    find gradient(in korean, 기울기 또는 미분값) for whole images.
    
    Arguments:
    ----------------------
    - images: images. (60000, 28, 28)
    
    Returns:
    ----------------------
    - grad: y-axis-oriented gradient (in korean, y 축 방향 기울기)
    """
    
    # some useful variables.
    m = images.shape[0]
    width = images.shape[1]
    height = images.shape[2]
    
    # define placeholder to store gradients.
    y_grads = np.zeros((m, width - 2, height - 2))
    
    # slice image into small size window, then compute gradient.
    for w in range(1, width - 1):
        for h in range(1, height - 1):
            images_slice = images[:, w - 1 : w + 2, h - 1 : h + 2]
            y_grads[:, w - 1, h - 1] = y_gradient_slice(images_slice)
            
    return y_grads

def get_average_grads(x_grads, y_grads, grid=7):
    """
    After we get gradients, let's compute average of these gradient. I'will post pictures.
    compute partial gradients.
    
    Arguments:
    ---------------------
    - x_grads: pre-computed gradients for x-axis (60000, 28, 28)
    - y_grads: pre-computed gradients for y-axis (60000, 28, 28)
    - grid: grid for dividing images. we will compute average of gradients for each grid. the averages become features.
    
    Returns:
    ---------------------
    - x_avg_grads: average of gradients x-axis (60000, 7, 7)
    - y_avg_grads: average of gradients y-axis (60000, 7, 7)
    """
    
    assert(x_grads.shape == y_grads.shape)
    
    # some useful variables.
    m = x_grads.shape[0]
    width = x_grads.shape[1]
    height = x_grads.shape[2]
    
    # I define these variables to slicing images conveniently.
    w_step = width // grid  # w_step = 4
    h_step = height // grid # h_step = 4
    
    # placeholder for storing average of gradients
    x_avg_grads = np.zeros((m, width // w_step, height // h_step))
    y_avg_grads = np.zeros((m, width // w_step, height // h_step))
    
    for w in range(0, width, w_step):
        for h in range(0, height, h_step):
            # slicing gradients into small part.
            x_grads_slice = x_grads[:, w : w + w_step, h : h + h_step]
            y_grads_slice = y_grads[:, w : w + w_step, h : h + h_step]
            
            assert(x_grads_slice.shape == y_grads_slice.shape == (m, width // grid, height // grid))
            
            # compute mean of gradients of part of image
            x_avg_grads[:, w // w_step, h // h_step] = np.mean(x_grads_slice, axis=(1, 2))
            y_avg_grads[:, w // w_step, h // h_step] = np.mean(y_grads_slice, axis=(1, 2))
            
    return x_avg_grads, y_avg_grads

def average_gradients_grid(images, grid=7, padding=1, normalize=True):
    """
    Preprocessing method 1 which I tried.
    
    Arguments:
    -------------------------
    - images: training or test images (60000, 28, 28)
    - grid: grid for dividing images. we will compute average of gradients for each grid. the averages become features.
    - padding: how much padding image.
    
    Returns:
    -------------------------
    - features: pre-processed features (pixel of images). (60000, 98)
    """
    
    images = np.copy(images)
    
    m = images.shape[0]
    
    # normalize
    if normalize:
        images_norm = images / 255
    else:
        images_norm = images
    
    # thresholding
    images_norm[images_norm >= 0.1] = 1
    images_norm[images_norm < 0.1] = 0
    
    # zero padding
    images_padded = zero_padding(images_norm, padding)
    
    # number of features = grid^2 * 2
    features = np.zeros((m, (grid ** 2) * 2))

    # compute x-axis gradient, y-axis gradient
    x_grads = x_gradient(images_padded)
    y_grads = y_gradient(images_padded)
    
    # compute average of gradient (grid 7x7)
    x_avg_grads, y_avg_grads = get_average_grads(x_grads, y_grads, grid)
    
    assert(x_avg_grads.shape == y_avg_grads.shape == (m, grid, grid))
    
    # flatten
    x_features = x_avg_grads.reshape(m, -1)
    y_features = y_avg_grads.reshape(m, -1)
    
    features[:, : grid ** 2] = x_features
    features[:, grid ** 2 :] = y_features
    
    return features

def apply_hog(images_train, images_test):
    m_tr = images_train.shape[0]
    m_ts = images_test.shape[0]
    
    # compute HoG (Historgram of Gradients)
    hog_train = np.zeros((m_tr, 81))
    hog_test = np.zeros((m_ts, 81))
    
    for i in range(m_tr):
        hog_train[i] = hog(images_train[i], block_norm='L2-Hys')
    for i in range(m_ts):
        hog_test[i] = hog(images_test[i], block_norm='L2-Hys')
        
    return hog_train, hog_test

def poly_model(grads, hogs, poly_degree):
    """
    This function reads inputs (grads, hogs) and then append into one vector.
    And, most importantly, make the features (grads + hogs vector) polynomial or exponential
    This has an effect that makes algorithm be applied to non-linear-separatable dataset.
    
    Arguments
    ---------------------------------
    grads: features containing gradients of images
    hogs: features containing histogram of gradients
    
    Returns
    ---------------------------------
    f_train: preprocessed features of training dataset this function generate
    f_test: preprocessed features of test dataset this function generate
    """
    
    # get number of training set (60000), number of test set (10000)
    m_train = grads[0].shape[0] # 60000
    m_test = grads[1].shape[0]  # 10000
    
#     # placeholder for new features.
#     f_train = np.zeros((m_train, (98 + 81) * poly_degree))
#     f_test = np.zeros((m_test, (98 + 81) * poly_degree)) 
    
    grads_train, grads_test = grads[0], grads[1]
    hogs_train, hogs_test = hogs[0], hogs[1]
    
#     for i in range(poly_degree):
#         f_train[:, (98 + 81) * i : (98 + 81) * i + 98] = grads_train ** (i + 1)
#         f_train[:, (98 + 81) * i + 98 : (98 + 81) * (i + 1)] = hogs_train ** (i + 1)
#         f_test[:, (98 + 81) * i : (98 + 81) * i + 98] = grads_test ** (i + 1)
#         f_test[:, (98 + 81) * i + 98 : (98 + 81) * (i + 1)] = hogs_test ** (i + 1)
        
    # 97.0
    f_train = np.zeros((m_train, (98 + 81) * poly_degree + 49 + 168 * 2))
    f_test = np.zeros((m_test, (98 + 81) * poly_degree + 49 + 168 * 2))
    
    
#     f_train = np.zeros((m_train, 98 * poly_degree + 168 * 2))
#     f_test = np.zeros((m_test, 98 * poly_degree + 168 * 2))
    
    for i in range(poly_degree):
        f_train[:, 98 * i : 98 * (i + 1)] = grads_train ** (i + 1)
        f_test[:, 98 * i : 98 * (i + 1)] = grads_test ** (i + 1)
    
    cur_w = 0
    cur_h = 0
    cur_idx = 98 * poly_degree
    
    while True:
        if cur_w == 7:
            cur_w = 0
            cur_h += 1
        if cur_h == 7:
            break
            
        # print(cur_h, cur_w)        
        
        if cur_w < 6:
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2], grads_train[:, cur_h * 14 + cur_w * 2 + 2])
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2], grads_test[:, cur_h * 14 + cur_w * 2 + 2])
            cur_idx += 1
            
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2 + 1], grads_train[:, cur_h * 14 + cur_w * 2 + 3])
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2 + 1], grads_test[:, cur_h * 14 + cur_w * 2 + 3])
            cur_idx += 1
            
        if cur_h < 6:
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2], grads_train[:, cur_h * 14 + cur_w * 2 + 14])
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2], grads_test[:, cur_h * 14 + cur_w * 2 + 14])
            cur_idx += 1
            
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2 + 1], grads_train[:, cur_h * 14 + cur_w * 2 + 15])
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2 + 1], grads_test[:, cur_h * 14 + cur_w * 2 + 15])
            cur_idx += 1
            
        cur_w += 1
            
    assert(cur_idx == 98 * poly_degree + 168)
    
    
    cur_w = 0
    cur_h = 0
    cur_idx = 98 * poly_degree + 168
    
    while True:
        if cur_w == 7:
            cur_w = 0
            cur_h += 1
        if cur_h == 7:
            break
            
        # print(cur_h, cur_w)        
        
        if cur_w < 6:
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2]**2, grads_train[:, cur_h * 14 + cur_w * 2 + 2]**2)
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2]**2, grads_test[:, cur_h * 14 + cur_w * 2 + 2]**2)
            cur_idx += 1
            
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2 + 1]**2, grads_train[:, cur_h * 14 + cur_w * 2 + 3]**2)
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2 + 1]**2, grads_test[:, cur_h * 14 + cur_w * 2 + 3]**2)
            cur_idx += 1
            
        if cur_h < 6:
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2]**2, grads_train[:, cur_h * 14 + cur_w * 2 + 14]**2)
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2]**2, grads_test[:, cur_h * 14 + cur_w * 2 + 14]**2)
            cur_idx += 1
            
            f_train[:, cur_idx] = np.multiply(grads_train[:, cur_h * 14 + cur_w * 2 + 1]**2, grads_train[:, cur_h * 14 + cur_w * 2 + 15]**2)
            f_test[:, cur_idx] = np.multiply(grads_test[:, cur_h * 14 + cur_w * 2 + 1]**2, grads_test[:, cur_h * 14 + cur_w * 2 + 15]**2)
            cur_idx += 1
            
        cur_w += 1
        
    assert(cur_idx == 98 * poly_degree + 168*2)
    
    f_train[:, cur_idx : cur_idx + 49] = np.multiply(grads_train[:, 0 :: 2], grads_train[:, 1 :: 2])
    f_test[:, cur_idx : cur_idx + 49] = np.multiply(grads_test[:, 0 :: 2], grads_test[:, 1 :: 2])
    
    cur_idx += 49
    
    for i in range(poly_degree):
        f_train[:, cur_idx + 81 * i : cur_idx + 81 * (i + 1)] = hogs_train  ** (i + 1)
        f_test[:, cur_idx + 81 * i : cur_idx + 81 * (i + 1)] = hogs_test ** (i + 1)
        
    cur_idx += 81 * poly_degree
    
    ### 97.0
        
    return f_train, f_test

def make_poly_features(images_train, images_test, poly_degree=3):
    """
    Arguments
    -------------------------
    images_train: images in training datase shaped of (60000, 28, 28)
    images_test: images in test dataset shaped of (10000, 28, 28)
    poly_degree: how many do you product exponentialy?
    
    Returns
    -------------------------
    f_train: new features preprocessed
    f_test: new features of test images.
    """
    
    m_train = images_train.shape[0]
    m_test = images_test.shape[0]
    
    # compute average of gradients
    avg_grads_train = average_gradients_grid(images_train)
    avg_grads_test = average_gradients_grid(images_test)
    
    hog_train, hog_test = apply_hog(images_train, images_test)
        
    f_train, f_test = poly_model((avg_grads_train, avg_grads_test), (hog_train, hog_test), poly_degree)
    
    return f_train, f_test