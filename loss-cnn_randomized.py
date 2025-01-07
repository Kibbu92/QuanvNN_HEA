
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
        x = jax.nn.relu(x)
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
    
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 10 
    BATCH_SIZE = 32
    DATSET_NAME = 'MNIST' 
    SPLIT_FACTOR = 0.3
    SHRINK_FACTOR = 2 
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
    
    acc_final=[]    
    for j in range(len(KNL)):
        
        KERNEL_SIZE = KNL[j]
        
        ii = 0
        for h in range(len(OPT)):
            
            optimizer = eval('optax.'+OPT[h]+'(LEARNING_RATE)')
            
            print(f'Test: {OPT[h]}')  
            
            SAVE_PATH = os.path.join('results', DATSET_NAME + '_CNN_Random2',  f'{KERNEL_SIZE[0]*KERNEL_SIZE[1]}_{KERNEL_SIZE}' , f'{OPT[h]}_{KERNEL_SIZE[0]*KERNEL_SIZE[1]}_{KERNEL_SIZE}')
            os.makedirs(SAVE_PATH, exist_ok=True)

            for m in range(5):
               
                rng_key = jax.random.PRNGKey(42) + 19000*m  
               
                forward = hk.transform(forward_fun)        
                params = forward.init(rng=rng_key, x=X_TRAIN[:BATCH_SIZE])              
               
                trainable_params = dict(params)
                non_trainable_params = {"CNN/~/C2D": trainable_params.pop("CNN/~/C2D")} # full
                                                       
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
                ACC = {'Acc_Train': acc_train_trajectory, 'Acc_Test': acc_test_trajectory, 'Loss': loss_trajectory}   
                
                with open(os.path.join(SAVE_PATH, 'Accuracy'+f'_{m}'+'.pkl'), 'wb') as f:
                    pickle.dump(ACC, f)                                               
   
                #===========================================================================                    
                Par_Grad = {'Par': param_trajectory, 'Grad': grad_trajectory}                   

                with open(os.path.join(SAVE_PATH, 'Par_Grad'+f'_{m}'+'.pkl'), 'wb') as f:
                    pickle.dump(Par_Grad, f)
                acc_final.append(acc_train_trajectory[-1])
    print('ECCOLO',np.mean(ACC))
                
          
                            
 
                                   
    
      
                    
     
      
     
        
 