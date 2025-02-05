
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

from keras.datasets.mnist import load_data
from keras.datasets.fashion_mnist import load_data as load_fashion_mnist
from keras.datasets.cifar10 import load_data as load_cifar

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
        
        print(np.unique(y))
        # Filter X to include only classes 0, 1 and 2
        class_0_1_indices = np.where((y == 0) | (y == 5) | (y == 9))[0]
        X = X[class_0_1_indices]
        y = y[class_0_1_indices]
        
        y = y-6

    X = X/X.max()

    if shrink_factor >1: 
        X = X[::shrink_factor, ...]
        y = y[::shrink_factor, ...]
        
        
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_factor, random_state=42)

    return X_train, X_test, y_train, y_test

#==============================================================================
#==============================================================================
def forward_fun(x):
    cnn = CNN(KNL=KERNEL_SIZE, NC=NUM_CLASSES)
    x = cnn(x)
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
    
    
    OPT =['adam']#, 'rmsprop', 'adagrad', 'lion', 'sm3', 'sgd', 'yogi', 'fromage', 'adabelief']        # MIGLIORI  'adam', 'rmsprop', 'lion', 'adabelief'
    KNL = [(2, 2, 3)]                 # [(2, 2, 3),(3, 3, 3),(4, 4, 3)] Dimensione Kernel             

    #===========================================================================
    # Load Dataset
    X_TRAIN, X_test, y_TRAIN, y_test = load_dataset(DATSET_NAME, SPLIT_FACTOR, SHRINK_FACTOR)
    print(f'Dataset Size - Train {X_TRAIN.shape} - Test {X_test.shape}')


    X_tot = np.concatenate((X_TRAIN,X_test),axis=0)
    y_tot = np.concatenate((y_TRAIN,y_test),axis=0)

    SCAN = []
    PT = []

    accuracies = []
    std_accuracies = []
    dataset_percentages = [1,5,10]+list(range(20,110,20))

    for percentage in dataset_percentages:  # From 10% to 100% in steps of 10
        print(f'Using {percentage}% of the dataset')
        
        mean_acc = []

        for dataset_seed in range(5):
            print(f'Dataset seed: {dataset_seed}')
            np.random.seed(dataset_seed)
        
            subset_size_X_TRAIN = int(len(X_TRAIN) * percentage / 100)
            random_indices = np.random.choice(range(len(X_TRAIN)), size=subset_size_X_TRAIN, replace=False)
            X_TRAIN_REDUCED = X_TRAIN[random_indices]
            y_TRAIN_REDUCED = y_TRAIN[random_indices]
        
            for j in range(len(KNL)):
                
                KERNEL_SIZE = KNL[j]
                
                ii = 0
                for h in range(len(OPT)):
                    
                    optimizer = eval('optax.'+OPT[h]+'(LEARNING_RATE)')
                    
                    print(f'Test: {OPT[h]}')  
                    
                    SAVE_PATH = os.path.join('results', DATSET_NAME + '_CNN_Reduced4',  f'{KERNEL_SIZE[0]*KERNEL_SIZE[1]}_{KERNEL_SIZE}' , f'{OPT[h]}_{KERNEL_SIZE[0]*KERNEL_SIZE[1]}_{KERNEL_SIZE}')
                    os.makedirs(SAVE_PATH, exist_ok=True)

                    mean_acc = []
                    for m in range(5):
                       
                        rng_key = jax.random.PRNGKey(42) + 19000*m  
                       
                        forward = hk.transform(forward_fun)        
                        params = forward.init(rng=rng_key, x=X_TRAIN_REDUCED[:BATCH_SIZE])              
                       
                        trainable_params = dict(params)
                        non_trainable_params = {} # full
                                                               
                        opt_state = optimizer.init(trainable_params)                
                        #===========================================================================
                        # Optimization Loop
                        loss_trajectory = []
                        param_trajectory = []
                        grad_trajectory = []
                        acc_train_trajectory = []
                        acc_test_trajectory = []
                        
                        X_train = X_TRAIN_REDUCED
                        y_train = y_TRAIN_REDUCED

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

    # Plotting the accuracy score against the dataset percentage
    plt.plot(dataset_percentages, accuracies, marker='o')
    plt.xlabel('Percentage of Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Dataset Percentage')
    plt.grid(True)
    plt.show()
    
    # open a file, where you stored the pickled data
    with open(os.path.join(SAVE_PATH, 'Mean_accuracies'+f'_{m}'+'.pkl'), 'rb') as f:
       accuracies=pickle.load(f)

    # Plotting the accuracy score against the dataset percentage
    plt.plot(dataset_percentages, accuracies, marker='o')
    plt.xlabel('Percentage of Dataset with loaded data')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Dataset Percentage')
    plt.grid(True)
    plt.show()
      
                 
                  
                 
                    
