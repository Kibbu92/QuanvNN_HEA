# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:18:41 2023

@author: andca
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import optax

def flat_traj(param_trajectory):
    
    b = []
    w = []
    q = []
    
    for i in range(len(param_trajectory)):
        
        b.append(param_trajectory[i]['full_1']['b'])
        w.append(param_trajectory[i]['full_1']['w'])
        q.append(param_trajectory[i]['qcnn']['angles'])
    
    b = np.asanyarray(b).reshape((np.shape(b)[0],-1))
    w = np.asanyarray(w).reshape((np.shape(w)[0],-1))
    q = np.asanyarray(q).reshape((np.shape(q)[0],-1))
    
    pt = np.concatenate((b,w,q),axis=1)
    
    return pt

if __name__ == '__main__':

    
    Dataset = 'MNIST_Epoch50'   #  MNIST_Epoch100
     
    folder = os.getcwd() + '\\results\\' + Dataset
 
    Layer_Krn = os.listdir(folder)
    Acc_train = []
    Acc_test  = []
    for i in range(len(Layer_Krn)):
        
        legn = []    

        fig,ax = plt.subplots()
        fig1,ax1 = plt.subplots(nrows = 2, ncols = 1)
        fig2,ax2 = plt.subplots(nrows = 2, ncols = 1)

        Folder_i = folder+ '\\' + Layer_Krn[i]
        OPT = os.listdir(Folder_i)  
        
        acc_train = np.zeros((len(OPT)))
        acc_test  = np.zeros((len(OPT)))
        
        for k in range(len(OPT)):
        
            Folder_opt = Folder_i + '/' + OPT[k] 
        
            with open(Folder_opt + '/accuracy.npy','rb') as f:
                acc_train_traj = np.load(f)   
                acc_test_traj  = np.load(f)
                acc_tot_traj   = np.load(f)
            
            with open(Folder_opt +'/Paremeter_Trajectories.npy','rb') as f:
                cost_params = np.load(f,allow_pickle=True) 
                param_traj  = np.load(f,allow_pickle=True) 
                loss_traj   = np.load(f,allow_pickle=True) 
                grad_traj   = np.load(f,allow_pickle=True) 
            
            #===========================================================================
            acc_train[k] = acc_train_traj[-1]
            acc_test[k]  = acc_test_traj[-1]
            #===========================================================================
            sh_b = param_traj[0]['full_1']['b'].shape
            sh_w = param_traj[0]['full_1']['w'].shape
            sh_q = param_traj[0]['qcnn']['angles'].shape
            
            len_b = sh_b[0]
            len_w = sh_w[0]*sh_w[1]
            len_q = sh_q[0]*sh_q[1]
            
            pt = flat_traj(param_traj)
            gt = flat_traj(grad_traj)
            
            #===========================================================================
            NORM = np.linalg.norm(gt[::5,len_b+len_w:],axis= 1)
            # NORM = np.mean(gt[:,len_b+len_w:],axis= 1)

            ax2[0].plot(NORM)    
            
            NORM = np.linalg.norm(gt[::5,0:len_b+len_w],axis= 1)  
            # NORM = np.mean(gt[:,0:len_b+len_w],axis= 1)

            ax2[1].plot(NORM)     
            

            #===========================================================================
            COS_dis = optax.cosine_distance(pt[0,len_b+len_w:], pt[:,len_b+len_w:])            
            ax1[0].plot(COS_dis)
            
            COS_dis = optax.cosine_distance(pt[0,0:len_b+len_w], pt[:,0:len_b+len_w])            
            ax1[1].plot(COS_dis)            
            #===========================================================================
            # Plot Acc/Loss Trends
            legn.append(  OPT[k].replace(Layer_Krn[i],"")[0:-1] )
            
            
            ax.plot(acc_test_traj)
        
            
        Acc_train.append(acc_train)
        Acc_test.append(acc_test)

        sz = 15        
        
        #===========================================================================
        ax2[0].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        ax2[0].set_ylabel('Mean Gradient Quantum Angles', fontsize=sz) 
        ax2[0].tick_params(axis='both', labelsize=sz)
        ax2[0].legend(legn)

        ax2[1].set_xlabel('Epoch', fontsize=sz) 
        ax2[1].set_ylabel('Mean Gradient Weights and Bias', fontsize=sz) 
        ax2[1].tick_params(axis='both', labelsize=sz)
        
        #===========================================================================
        ax1[0].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        ax1[0].set_ylabel('Cosine Distance Quantum Angles', fontsize=sz) 
        ax1[0].tick_params(axis='both', labelsize=sz)
        ax1[0].legend(legn)

        ax1[1].set_xlabel('Epoch', fontsize=sz) 
        ax1[1].set_ylabel('Cosine Distance Weights and Bias', fontsize=sz) 
        ax1[1].tick_params(axis='both', labelsize=sz)

        #===========================================================================
        ax.legend(legn)
        ax.set_xlabel('Epoch', fontsize=sz) 
        ax.set_ylabel('Test Accuracy', fontsize=sz) 
        ax.tick_params(axis='both', labelsize=sz)
        ax.set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
 
        
   


   
