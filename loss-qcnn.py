from orqviz.pca import (get_pca, perform_2D_pca_scan, plot_pca_landscape, 
                        plot_optimization_trajectory_on_pca)
# from orqviz.scans import    plot_2D_scan_result_as_3D
    
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.utils import gen_batches
from sklearn.utils import shuffle
from pennylane import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import haiku as hk
# import pickle
import glob
import optax
import jax
import cv2
import os

# from scipy.interpolate import interpn

from QCNN import *


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

def forward_fun(x):
    qcnn = QCNN(kernel_size=KERNEL_SIZE, n_layers=NUM_LAYERS)
    full1 = hk.Linear(NUM_CLASSES, name="full_1")
    n_samples = len(x)

    x = qcnn(x)
    
    x = x.reshape((n_samples, -1))
    x = full1(x)
    return x

def lossFn(trainable_params: hk.Params, non_trainable_params: hk.Params, images, labels):
    """Cross-entropy classification loss"""
    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    logits = forward.apply(params,rng_key, images)
    result = optax.softmax_cross_entropy_with_integer_labels(logits, labels).sum()
    return result

@jax.jit
def evaluate(trainable_params: hk.Params, non_trainable_params: hk.Params, images, labels) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    logits = forward.apply(params,rng_key, images)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

@jax.jit
def update(opt_state, trainable_params, non_trainable_params, images, labels):
    loss, grads = jax.value_and_grad(lossFn)(trainable_params, non_trainable_params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
    trainable_params = optax.apply_updates(trainable_params, updates)
    return trainable_params, opt_state, loss, grads

def params_to_flat_array(param_trajectory):
    result = np.array([0.,])
    result = np.concatenate((result, np.array(param_trajectory['full_1']['b']).flatten(),))
    result = np.concatenate((result, np.array(param_trajectory['full_1']['w']).flatten(),))
    result = np.concatenate((result, np.array(param_trajectory['qcnn']['angles']).flatten(),))
    return np.array(result).flatten()[1:]

def get_params_from_flat_array(flat_array):
    len_b = shape_b[0]
    len_w = shape_w[0]*shape_w[1]

    b_raw = flat_array[:len_b].reshape(shape_b)
    w_raw = flat_array[len_b:len_b+len_w].reshape(shape_w)
    angles_raw = flat_array[len_b+len_w:].reshape(shape_angles)
    
    # params = {'qcnn': {'angles': angles_raw}, 'full_1': {'b': b_raw, 'w': w_raw}}
    params = {'qcnn': {'angles': angles_raw}, 'full_1': {'b': b_raw, 'w': w_raw}}
    return params

@jax.jit
def partCostFn(flat_params):
    params = get_params_from_flat_array(flat_params)
    return lossFn(params, non_trainable_params, X_train, y_train)


if __name__ == '__main__':
    
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 10 
    BATCH_SIZE = 32
    DATSET_NAME = 'MNIST' 
    SPLIT_FACTOR = 0.3
    SHRINK_FACTOR = 1 
    EPOCHS = 100
    
    OPT = ['fromage'] #('adam', 'rmsprop', 'adagrad', 'lion', 'sm3', 'sgd', 'yogi', 'fromage', 'adabelief') # 
    NL  = [30] 
    KNL = [(2, 2, 3)] 
    
    rng_key = jax.random.PRNGKey(42)


    #===========================================================================
    # Load Dataset
    X_train, X_test, y_train, y_test = load_dataset(DATSET_NAME, SPLIT_FACTOR, SHRINK_FACTOR)
    print(f'Dataset Size - Train {X_train.shape} - Test {X_test.shape}')
    
    X_tot = np.concatenate((X_train,X_test),axis=0)
    y_tot = np.concatenate((y_train,y_test),axis=0)
    #===========================================================================
    # Define Forward


    for h in range(len(OPT)):
        
        optimizer = eval('optax.'+OPT[h]+'(LEARNING_RATE)')
        
        print(f'Test: {OPT[h]}')

        for k in range(len(NL)):
            
            NUM_LAYERS = NL[k]
            
            for j in range(len(KNL)):
                
                KERNEL_SIZE = KNL[j]
            
                forward = hk.transform(forward_fun)
        
                params = forward.init(rng=rng_key, x=X_train[:BATCH_SIZE])
                
                SAVE_PATH = os.path.join('results', DATSET_NAME, f'{OPT[h]}_{NUM_LAYERS}_{KERNEL_SIZE}')
                os.makedirs(SAVE_PATH, exist_ok=True)
                
                trainable_params = dict(params)
                non_trainable_params = {"qcnn": {"gates": trainable_params["qcnn"].pop("gates")}} 
                
                opt_state = optimizer.init(trainable_params)
                #===========================================================================
                # Optimization Loop
                loss_trajectory = []
                param_trajectory = [trainable_params, ]
                grad_trajectory = []
                acc_train_trajectory = []
                acc_test_trajectory = []
                acc_tot_trajectory = []
            
                for i in range(EPOCHS):
                    X_train, y_train = shuffle(X_train, y_train, random_state=i)
                    batch_slices = gen_batches(len(X_train), BATCH_SIZE)
            
                    for batch in batch_slices:
                        trainable_params, opt_state, loss_value, grads = update(opt_state, 
                                                                                trainable_params, 
                                                                                non_trainable_params, 
                                                                                X_train[batch], y_train[batch])
                        
                        loss_trajectory.append(loss_value)
                        param_trajectory.append(trainable_params)
                        grad_trajectory.append(grads)
            
                    if i % 5 == 0:
                        acc_train = evaluate(trainable_params, non_trainable_params, X_train, y_train)
                        acc_test = evaluate(trainable_params, non_trainable_params, X_test, y_test)
                        acc_tot = evaluate(trainable_params, non_trainable_params, X_tot, y_tot)
            
                        acc_train_trajectory.append(acc_train)
                        acc_test_trajectory.append(acc_test)
                        acc_tot_trajectory.append(acc_tot)
            
                        print(f'step {i}, loss: {loss_value}, ACC-train: {acc_train}, ACC-test: {acc_test}')
                
                acc_train = evaluate(trainable_params, non_trainable_params, X_train, y_train)
                acc_test = evaluate(trainable_params, non_trainable_params, X_test, y_test)
                
               
            
                
                with open(os.path.join(SAVE_PATH, 'accuracy.npy'), 'wb') as f:
                    np.save(f, acc_train_trajectory)
                    np.save(f, acc_test_trajectory)
                    np.save(f, acc_tot_trajectory) 
            
                    
                with open(os.path.join(SAVE_PATH, 'Paremeter_Trajectories.npy'),'wb') as f:
                    np.save(f, non_trainable_params)  
                    np.save(f, param_trajectory)
                    np.save(f, loss_trajectory)
                    np.save(f, grad_trajectory) 
      

     
        
    # with open(os.path.join(SAVE_PATH, 'acc.txt'), 'w') as f:
    #     f.write(f'ACC Train {acc_train} \n')
    #     f.write(f'ACC Test  {acc_test}  \n')
    #     f.write(f'ACC Tot   {acc_tot}   \n')

    # #===========================================================================
    # # Plot Acc/Loss Trends

    # fig, ax = plt.subplots(nrows = 1, ncols = 2)
    # ax[0].plot(acc_train_trajectory)
    # ax[0].plot(acc_test_trajectory)
    # ax[0].legend(['Train', 'Test'])
    # ax[0].set_title('Accuracy')

    # ax[1].plot(loss_trajectory)
    # ax[1].set_title('Loss')
    # plt.show()

    # fig.tight_layout()
    # plt.savefig(os.path.join(SAVE_PATH, 'acc-loss.png'))
    # plt.close()

    # #===========================================================================
    # # Orqviz scanning
    # shape_b = param_trajectory[0]['full_1']['b'].shape
    # shape_w = param_trajectory[0]['full_1']['w'].shape
    # shape_angles = param_trajectory[0]['qcnn']['angles'].shape
    # flat_array = params_to_flat_array(param_trajectory[0])
    # param_dict = get_params_from_flat_array(flat_array)
    # flat_param_trajectory = [params_to_flat_array(param_trajectory_item) for param_trajectory_item in param_trajectory]
    # pt = np.array(flat_param_trajectory)
    # pca = get_pca(pt)
    # scan_pca_result = perform_2D_pca_scan(pca, partCostFn, n_steps_x=20, offset=5)
    


    # fig, ax = plt.subplots()
    # plot_pca_landscape(scan_pca_result, pca, fig=fig, ax=ax)
    # plot_optimization_trajectory_on_pca(pt, pca, ax=ax, 
    #                                     label="Optimization Trajectory", color="lightsteelblue")
    # ax.legend()
    # plt.show()
    





