
# from orqviz.pca import get_pca, perform_2D_pca_scan

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.utils import gen_batches
from sklearn.utils import shuffle
from pennylane import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import haiku as hk
import glob
import optax
import jax
import cv2
import os
import pickle
from skimage import util
from keras.datasets.mnist import load_data
from skimage.filters import gaussian
from keras.datasets.mnist import load_data
from keras.datasets.fashion_mnist import load_data as load_fashion_mnist
from keras.datasets.cifar10 import load_data as load_cifar

from QCNN import *



###############################################################################
###############################################################################
class CNN(hk.Module):
    def __init__(self,  KNL, NC):
        super().__init__(name='CNN')
        self.conv = hk.Conv2D(output_channels=KNL[0]*KNL[1], kernel_shape=KNL[0:2], stride=1,name='C2D', padding="VALID")
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(NC,name='FULL')

    def __call__(self, x):
        x = self.conv(x)
        #x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
###############################################################################
################################# FUNCTIONS ###################################
###############################################################################
           
def load_dataset(dataset_name, split_factor=0.3, shrink_factor=12):

    if dataset_name=='EuroSAT':
        X, y = [], []

        for j, paths_class in enumerate(glob.glob(os.path.join('EuroSAT', '*'))):
            for img in glob.glob(os.path.join(paths_class, '*')):
                i = plt.imread(img)
                i = i[:,:,:3]/255.0
                i = cv2.resize(i, (8, 8)) 
                X.append(i)
                y.append(j)

        X = np.array(X)
        y = np.array(y)

    elif dataset_name == 'MNIST':
        X, y =  load_digits(return_X_y=True)
        X = X.repeat(3).reshape((-1, 8, 8, 3))
        
    elif dataset_name == 'MNIST_FULL':
        train_data, test_data =  load_data()
        X = np.concatenate((train_data[0],test_data[0]),axis=0)
        y = np.concatenate((train_data[1],test_data[1]),axis=0)
        X = X.repeat(3).reshape((-1, 28, 28, 3))
        
    elif dataset_name == 'FASHION_MNIST_FULL':
        train_data, test_data =  load_fashion_mnist()
        X = np.concatenate((train_data[0],test_data[0]),axis=0)
        y = np.concatenate((train_data[1],test_data[1]),axis=0)
        X = X.repeat(3).reshape((-1, 28, 28, 3))
        
        # Filter X to include only classes 0, 1 and 2
        class_0_1_indices = np.where((y == 0) | (y == 1) | (y == 2))[0]
        X = X[class_0_1_indices]
        y = y[class_0_1_indices]
       
    elif dataset_name == 'CIFAR':
        train_data, test_data =  load_cifar()
        X = np.concatenate((train_data[0],test_data[0]),axis=0)
        y = np.concatenate((train_data[1],test_data[1]),axis=0)
        
        # Filter X to include only classes 0, 1 and 2
        class_0_1_indices = np.where((y == 0) | (y == 1))[0]
        X = X[class_0_1_indices]
        y = y[class_0_1_indices]
        
        X = X.reshape((-1, 32, 32, 3))

    X = X/X.max()

    if shrink_factor >1: 
        X = X[::shrink_factor, ...]
        y = y[::shrink_factor, ...]
        
        
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_factor, random_state=42)

    return X_train, X_test, y_train.reshape(-1,), y_test.reshape(-1,)
    

# Function to add Gaussian noise to images
def add_gaussian_noise(image, sigma=0.1):
    noisy_image = util.random_noise(image, mode='gaussian', var=sigma**2)
    return noisy_image

# Function to add black rectangles/bands to images
def add_black_rectangles(image, num_rectangles=3):
    rect_width = int(image.shape[0]/5)
    rect_height = int(image.shape[0]/5)
    res_image = image.copy()
    for _ in range(num_rectangles):
        x = np.random.randint(0, image.shape[0] - rect_height)
        y = np.random.randint(0, image.shape[1] - rect_width)
        res_image[x:x+rect_height, y:y+rect_width] = 0
    return res_image

# Function to apply Gaussian blur to images
def apply_gaussian_blur2(image, sigma=1):
    blurred_image = gaussian(image, sigma=sigma, multichannel=True)
    return blurred_image
    
# Function to apply Gaussian blur to images
def apply_gaussian_blur(image, sigma=1):
    blurred_image = cv2.medianBlur(image.astype('float32'), 5)
    return blurred_image

#==============================================================================
#==============================================================================
def forward_fun(x):
    qcnn = QCNN(kernel_size=KERNEL_SIZE, n_layers=NUM_LAYERS)
    full1 = hk.Linear(NUM_CLASSES, name="full")
    n_samples = len(x)

    x = qcnn(x)
    
    x = x.reshape((n_samples, -1))
    x = full1(x)
    return x

#==============================================================================
#==============================================================================
def lossFn(trainable_params: hk.Params, non_trainable_params: hk.Params, images, labels):
    """Cross-entropy classification loss"""
    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    logits = forward.apply(params,rng_key, images)
    result = optax.softmax_cross_entropy_with_integer_labels(logits, labels).sum()
    return result

#------------------------------------------------------------------------------
@jax.jit
def evaluate(trainable_params: hk.Params, non_trainable_params: hk.Params, images, labels) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    logits = forward.apply(params,rng_key, images)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

#------------------------------------------------------------------------------
@jax.jit
def update(opt_state, trainable_params, non_trainable_params, images, labels):
    loss, grads = jax.value_and_grad(lossFn)(trainable_params, non_trainable_params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
    trainable_params = optax.apply_updates(trainable_params, updates)
    return trainable_params, opt_state, loss, grads

#==============================================================================
#==============================================================================
def params_to_flat_array(param_trajectory):
    result = np.array([0.,])
    result = np.concatenate((result, np.array(param_trajectory['full']['b']).flatten(),))
    result = np.concatenate((result, np.array(param_trajectory['full']['w']).flatten(),))
    result = np.concatenate((result, np.array(param_trajectory['qcnn']['angles']).flatten(),))
    return np.array(result).flatten()[1:]

#------------------------------------------------------------------------------
def get_params_from_flat_array(flat_array):
    len_b = shape_b[0]
    len_w = shape_w[0]*shape_w[1]

    b_raw = flat_array[:len_b].reshape(shape_b)
    w_raw = flat_array[len_b:len_b+len_w].reshape(shape_w)
    angles_raw = flat_array[len_b+len_w:].reshape(shape_q)
    
    # params = {'qcnn': {'angles': angles_raw}, 'full': {'b': b_raw, 'w': w_raw}}
    params = {'qcnn': {'angles': angles_raw}, 'full': {'b': b_raw, 'w': w_raw}}
    return params

#------------------------------------------------------------------------------
@jax.jit
def partCostFn(flat_params):
    params = get_params_from_flat_array(flat_params)
    return lossFn(params, non_trainable_params, X_train, y_train)

#------------------------------------------------------------------------------
###############################################################################


###############################################################################
#################################### MAIN #####################################
###############################################################################
if __name__ == '__main__':

    
    LEARNING_RATE = 1e-2
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    DATSET_NAME = 'FASHION_MNIST_FULL' 
    SPLIT_FACTOR = 0.3
    SHRINK_FACTOR = 1
    EPOCHS = 100
    
    # ['adam', 'rmsprop', 'adagrad', 'lion', 'sm3', 'sgd', 'yogi', 'fromage', 'adabelief']   
    
    OPT = ['adam']#['adam', 'rmsprop', 'adagrad', 'lion', 'sm3', 'sgd', 'yogi', 'fromage', 'adabelief']      # MIGLIORI  'adam', 'rmsprop', 'lion', 'adabelief'
    NL  = [13]                                       # [ 3, 10, 30] Numero di Layer
    KNL = [(2, 2, 3)]                        # [(2, 2, 3),(3, 3, 3),(4, 4, 3)] Dimensione Kernel   
    
    #===========================================================================
    # Load Dataset
    X_TRAIN, X_test, y_TRAIN, y_test = load_dataset(DATSET_NAME, SPLIT_FACTOR, SHRINK_FACTOR)
    print(f'Dataset Size - Train {X_TRAIN.shape} - Test {X_test.shape}')
    
    # Generate degraded versions of test images
    noisy_images_X_train = np.array([add_gaussian_noise(image) for image in X_TRAIN])
    noisy_images_X_test = np.array([add_gaussian_noise(image) for image in X_test])
    images_with_black_rectangles_X_train = np.array([add_black_rectangles(image) for image in X_TRAIN])
    images_with_black_rectangles_X_test = np.array([add_black_rectangles(image) for image in X_test])
    blurred_images_X_train = X_TRAIN#np.array([apply_gaussian_blur(image) for image in X_TRAIN])
    blurred_images_X_test = np.array([apply_gaussian_blur(image) for image in X_test])
    all_perturbations_images_X_train = np.array([add_black_rectangles(add_gaussian_noise(image)) for image in X_TRAIN])
    all_perturbations_images_X_test = np.array([add_black_rectangles(add_gaussian_noise(image)) for image in X_test])

    #===========================================================================
    # Show noisy image
    # Plot original and modified images side by side
    plt.figure(figsize=(10, 5))

    # Original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(X_TRAIN[4], cmap='gray')
    plt.title('Original Image')

    # Modified image on the right
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_images_X_train[4], cmap='gray')
    plt.title('Noisy Image')

    plt.show()
    
    #===========================================================================
    # Show black rectangle image
    # Plot original and modified images side by side
    plt.figure(figsize=(10, 5))

    # Original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(X_TRAIN[100], cmap='gray')
    plt.title('Original Image')

    # Modified image on the right
    plt.subplot(1, 2, 2)
    plt.imshow(images_with_black_rectangles_X_train[100], cmap='gray')
    plt.title('Noisy Image')

    plt.show()
    
    #===========================================================================
    # Show blurred image
    # Plot original and modified images side by side
    plt.figure(figsize=(10, 5))

    # Original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(X_TRAIN[2000], cmap='gray')
    plt.title('Original Image')

    # Modified image on the right
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_images_X_train[2000], cmap='gray')
    plt.title('Noisy Image')

    plt.show()
    
    #===========================================================================
    # Show all perturbations image
    # Plot original and modified images side by side
    plt.figure(figsize=(10, 5))

    # Original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(X_TRAIN[3000], cmap='gray')
    plt.title('Original Image')

    # Modified image on the right
    plt.subplot(1, 2, 2)
    plt.imshow(all_perturbations_images_X_train[3000], cmap='gray')
    plt.title('Noisy Image')

    plt.show()
    
    
    
    #X_tot = np.concatenate((X_TRAIN,X_test),axis=0)
    #y_tot = np.concatenate((y_TRAIN,y_test),axis=0)
    
    dataset_type = 'blur' #occlusion, blur
    
    if dataset_type=='noise':
        X_TRAIN = noisy_images_X_train #images_with_black_rectangles_X_train, blurred_images_X_train
        X_test  = noisy_images_X_test #images_with_black_rectangles_X_test, blurred_images_X_test 
    elif dataset_type=='occlusion':
        X_TRAIN = images_with_black_rectangles_X_train 
        X_test  = images_with_black_rectangles_X_test 
    elif dataset_type=='blur':
        X_TRAIN = blurred_images_X_train 
        X_test  = blurred_images_X_test 
    elif dataset_type=='all':
        X_TRAIN = all_perturbations_images_X_train 
        X_test  = all_perturbations_images_X_test


    SCAN = []
    PT = []
    accuracies = []
    std_accuracies = []
        
    for k in range(len(NL)):
        
        NUM_LAYERS = NL[k]
        
        for j in range(len(KNL)):
            
            KERNEL_SIZE = KNL[j]
            
            ii = 0
            for h in range(len(OPT)):
                
                optimizer = eval('optax.'+OPT[h]+'(LEARNING_RATE)')
                
                print(f'Test: {OPT[h]}')  
                
                SAVE_PATH = os.path.join('results', DATSET_NAME, dataset_type,  f'{NUM_LAYERS}_{KERNEL_SIZE}' , f'{OPT[h]}_{NUM_LAYERS}_{KERNEL_SIZE}')
                os.makedirs(SAVE_PATH, exist_ok=True)

                mean_acc = []
                for m in range(5):
                    
                    rng_key = jax.random.PRNGKey(130) + 19000*m  

                    forward = hk.transform(forward_fun)        
                    params = forward.init(rng=rng_key, x=X_TRAIN[:BATCH_SIZE])               
                      
                    trainable_params = dict(params)
                    non_trainable_params = {"qcnn": {"gates": trainable_params["qcnn"].pop("gates")}} 
                    
                    opt_state = optimizer.init(trainable_params)                
                    #===========================================================================
                    # Optimization Loop
                    loss_trajectory = []
                    param_trajectory = []
                    grad_trajectory = []
                    acc_train_trajectory = []
                    acc_test_trajectory = []
                    
                    X_train = X_TRAIN
                    y_train = y_TRAIN

                    for i in range(EPOCHS+1):
                        
                        X_train, y_train = shuffle(X_train, y_train, random_state=i)
                        batch_slices = gen_batches(len(X_train), BATCH_SIZE)
                        
                        if i==0:
                            loss_value, grads = jax.value_and_grad(lossFn)(trainable_params, non_trainable_params, X_train[:BATCH_SIZE], y_train[:BATCH_SIZE])
                            
                        if i % 5 == 0:
                            acc_train = evaluate(trainable_params, non_trainable_params, X_train, y_train)
                            acc_test = evaluate(trainable_params, non_trainable_params, X_test, y_test)
                
                            acc_train_trajectory.append(acc_train)
                            acc_test_trajectory.append(acc_test)               
                            print(f'step {i}, loss: {loss_value}, ACC-train: {acc_train}, ACC-test: {acc_test}') 

                        param_trajectory.append(trainable_params)
                        loss_trajectory.append(loss_value)
                        grad_trajectory.append(grads)

                        for batch in batch_slices:
                            trainable_params, opt_state, loss_value, grads = update(opt_state, 
                                                                                    trainable_params, 
                                                                                    non_trainable_params, 
                                                                                    X_train[batch], y_train[batch])  
                
                            
                    #===========================================================================
                    '''
                    ACC = {'Acc_Train': acc_train_trajectory, 'Acc_Test': acc_test_trajectory, 'Loss': loss_trajectory}

                    with open(os.path.join(SAVE_PATH, 'Accuracy'+f'_{m}'+'.pkl'), 'wb') as f:
                      pickle.dump(ACC, f)

                    #===========================================================================
                    Par_Grad = {'Par': param_trajectory, 'Grad': grad_trajectory}

                    with open(os.path.join(SAVE_PATH, 'Par_Grad'+f'_{m}'+'.pkl'), 'wb') as f:
                      pickle.dump(Par_Grad, f)
                    '''
                    mean_acc.append(acc_test_trajectory[-1])
                accuracies.append(np.mean(mean_acc))
                std_accuracies.append(np.std(mean_acc))
        with open(os.path.join(SAVE_PATH, 'Mean_accuracies'+f'_{m}'+'.pkl'), 'wb') as f:
            pickle.dump(accuracies, f)
        with open(os.path.join(SAVE_PATH, 'Std_accuracies'+f'_{m}'+'.pkl'), 'wb') as f:
            pickle.dump(std_accuracies, f)
        print(accuracies,std_accuracies)
          
                            
 
                                   
    
      
                    
     
      
     
        
 