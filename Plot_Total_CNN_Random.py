# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:18:41 2023

@author: andca
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import optax
from math import prod

from scipy.optimize import curve_fit
from orqviz.pca import plot_pca_landscape, plot_optimization_trajectory_on_pca


#==============================================================================
#==============================================================================
#==============================================================================

def flat_traj(param_trajectory):
    
    bFL = []
    wFL = []
    
    for i in range(len(param_trajectory)):
        
        bFL.append(param_trajectory[i]['CNN/~/FULL']['b'])
        wFL.append(param_trajectory[i]['CNN/~/FULL']['w'])    

    bFL = np.asanyarray(bFL).reshape((np.shape(bFL)[0],-1))
    wFL = np.asanyarray(wFL).reshape((np.shape(wFL)[0],-1))
    
    ptFL = np.concatenate((bFL,wFL),axis=1)

    return ptFL

def func(x, a, b, c, d, e, f):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e * np.exp(-f * x)    
#==============================================================================
#==============================================================================
#==============================================================================

if __name__ == '__main__':
    
    SS = 100
    ind = 0
    sz = 25 
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    
    folder = os.getcwd() + '\\results\\MNIST_CNN_Random'  #MNIST_CNN
 
    Layer_Krn = os.listdir(folder)
    
    PCA = []
    SCAN =[]
    PT_tot = []
    print('')
    #==========================================================================
    for i in range(len(Layer_Krn)):
        legn = []  
        print('')

        print(f'{Layer_Krn[i]}')
        print('')

        fig0,ax0 = plt.subplots(nrows = 3, ncols = 3, constrained_layout = True, figsize=(15, 8))
        fig1,ax1 = plt.subplots(nrows = 3, ncols = 3, constrained_layout = True, figsize=(15, 8))
        fig2,ax2 = plt.subplots(nrows = 3, ncols = 3, constrained_layout = True, figsize=(15, 8))
        fig3,ax3 = plt.subplots( figsize=(15, 13))
        fig4,ax4 = plt.subplots( figsize=(15, 13))
        
        
        Folder_i = folder+ '\\' + Layer_Krn[i]
        OPT = os.listdir(Folder_i)  
        legn = []
        
        PT = []
        #======================================================================
        for k in range(len(OPT)):

            text = f'{OPT[k].replace(Layer_Krn[i],"")[0:-1]}'
            
            legn.append(text)

            Folder_opt = Folder_i + '\\' + OPT[k] 
            
            File  = os.listdir(Folder_opt)   
            len_m = len(File)//2   

           
            Loss      = []    
            Acc_test  = []
            Acc_train = []  
            Gradient  = []  
            COS     = []
            DG      = []
            DG_C      = [] 
            
           
            Pt = []
            #==================================================================
            for m in range(len_m):
                
                with open(Folder_opt + '\\' + File[m], 'rb') as f:
                    ACC = pickle.load(f)  
                    
                Acc_test.append(ACC['Acc_Test'])
                Acc_train.append(ACC['Acc_Train'])
                Loss.append(ACC['Loss'])
                
                with open(Folder_opt + '\\' + File[len_m+m], 'rb') as f:
                    TMP = pickle.load(f) 
                
                pt = TMP['Par']    
                ptFL = flat_traj(pt)   
                
                COS_dis = optax.cosine_distance(ptFL[0,:], ptFL)
                
                COS.append(COS_dis)
                
                #-----------------------------------------
                gtFL = flat_traj(TMP['Grad'])

                Dg = np.sqrt(np.sum(gtFL**2,axis=1))
                
                DG.append(Dg)
                
                #-----------------------------------------
                if m==3:
                    pt_tmp = pt.copy()           
            
            #==================================================================  
            PT.append(pt_tmp)
            
            #------------------------------------------------------------------
            Epoch_acc = np.arange(0,101,5)     

            Acc_test_mean  = np.mean(Acc_test,axis=0)
            Acc_test_std   = np.std(Acc_test,axis=0)  
            
            print(text + '----------------------------------------------------')
            print(f'Accuracy: {Acc_test_mean[-1]} +- {Acc_test_std[-1]}')
            
            ax0[k//3,k-(k//3)*3].plot(Epoch_acc,Acc_test_mean,linestyle='--', marker='o',markersize=3, color='black') 
            ax0[k//3,k-(k//3)*3].fill(np.append(Epoch_acc,Epoch_acc[::-1]),np.append(Acc_test_mean+3*Acc_test_std,  Acc_test_mean[::-1]-3*Acc_test_std[::-1]), 
                                      color='gray',alpha=0.7)
            ax0[k//3,k-(k//3)*3].tick_params(axis='both', labelsize=sz)
            ax0[k//3,k-(k//3)*3].text(0.5, 0.3, text, transform=ax0[k//3,k-(k//3)*3].transAxes, fontsize=sz,
                    verticalalignment='top', bbox=props)
            ax0[k//3,k-(k//3)*3].grid()
            ax0[k//3,k-(k//3)*3].set_xlim([0, 100])
            ax0[k//3,k-(k//3)*3].set_ylim([0, 1.1])
            
            #------------------------------------------------------------------
            Acc_train_mean = np.mean(Acc_train,axis=0)
            Acc_train_std  = np.std(Acc_train,axis=0)
            
            ax1[k//3,k-(k//3)*3].plot(Epoch_acc,Acc_train_mean,linestyle='--', marker='o',markersize=3, color='black') 
            ax1[k//3,k-(k//3)*3].fill(np.append(Epoch_acc,Epoch_acc[::-1]),np.append(Acc_train_mean+3*Acc_train_std,  Acc_train_mean[::-1]-3*Acc_train_std[::-1]), 
                                      color='gray',alpha=0.7)
            ax1[k//3,k-(k//3)*3].tick_params(axis='both', labelsize=sz)
            ax1[k//3,k-(k//3)*3].text(0.5, 0.3, text, transform=ax0[k//3,k-(k//3)*3].transAxes, fontsize=sz,
                    verticalalignment='top', bbox=props)
            ax1[k//3,k-(k//3)*3].grid()
            ax1[k//3,k-(k//3)*3].set_xlim([0, 100])
            ax1[k//3,k-(k//3)*3].set_ylim([0, 1.1])
            
            #------------------------------------------------------------------
            Epoch_Loss = np.arange(0,101)
            Loss_mean = np.mean(Loss,axis=0)
            Loss_std  = np.std(Loss,axis=0)   
            
            popt_mean, pcov = curve_fit(func, Epoch_Loss, Loss_mean, maxfev = 2000000)
            popt_std, pcov = curve_fit(func, Epoch_Loss, Loss_std, maxfev = 2000000)

            Loss_mean = func(Epoch_Loss,*popt_mean)
            Loss_std  = func(Epoch_Loss,*popt_std)
            
            ax2[k//3,k-(k//3)*3].plot(Epoch_Loss,Loss_mean,linestyle='--', color='black') 
            ax2[k//3,k-(k//3)*3].fill(np.append(Epoch_Loss,Epoch_Loss[::-1]),np.append(Loss_mean+3*Loss_std,  Loss_mean[::-1]-3*Loss_std[::-1]), 
                                      color='gray',alpha=0.7)
            ax2[k//3,k-(k//3)*3].plot(Epoch_Loss[Epoch_acc],Loss_mean[Epoch_acc],linestyle='none', marker='o',markersize=3, color='black') 
            ax2[k//3,k-(k//3)*3].tick_params(axis='both', labelsize=sz)
            ax2[k//3,k-(k//3)*3].text(0.5, 0.85, text, transform=ax0[k//3,k-(k//3)*3].transAxes, fontsize=sz,
                    verticalalignment='top', bbox=props)
            ax2[k//3,k-(k//3)*3].grid()
            ax2[k//3,k-(k//3)*3].set_xlim([-1, 100])
            ax2[k//3,k-(k//3)*3].set_ylim([-1, 90])
            
            #------------------------------------------------------------------
            Cos_mean = np.mean(COS,axis=0)            
            ax3.plot(Epoch_Loss,Cos_mean,linewidth=3)                     
            
            #------------------------------------------------------------------
            DG_mean = np.mean(DG,axis=0)
            
            popt_q_mean, pcov = curve_fit(func, Epoch_Loss, DG_mean, maxfev = 2000000)

            DG_mean = func(Epoch_Loss,*popt_q_mean)
            
            ax4.plot(Epoch_Loss,DG_mean,linewidth=3)                     
            
        #======================================================================
        ax0[1,1].legend(["$\mu$", "3$\sigma$"]   )
        ax0[0,1].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        
        ax0[2,0].set_xlabel('Epoch', fontsize=sz) 
        ax0[2,1].set_xlabel('Epoch', fontsize=sz) 
        ax0[2,2].set_xlabel('Epoch', fontsize=sz)   
        
        ax0[0,0].set_ylabel('Test Accuracy', fontsize=sz) 
        ax0[1,0].set_ylabel('Test Accuracy', fontsize=sz) 
        ax0[2,0].set_ylabel('Test Accuracy', fontsize=sz) 
        
        #----------------------------------------------------------------------  
        ax1[1,1].legend(["$\mu$", "3$\sigma$"], fontsize=sz-10  )
        ax1[0,1].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        
        ax1[2,0].set_xlabel('Epoch', fontsize=sz) 
        ax1[2,1].set_xlabel('Epoch', fontsize=sz) 
        ax1[2,2].set_xlabel('Epoch', fontsize=sz)   
    
        ax1[0,0].set_ylabel('Train Accuracy', fontsize=sz) 
        ax1[1,0].set_ylabel('Train Accuracy', fontsize=sz) 
        ax1[2,0].set_ylabel('Train Accuracy', fontsize=sz) 
        
        #----------------------------------------------------------------------  
        ax2[1,1].legend(["$\mu$", "3$\sigma$"], fontsize=sz-10    )
        # ax2[0,1].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
    
        ax2[2,0].set_xlabel('Epoch', fontsize=sz) 
        ax2[2,1].set_xlabel('Epoch', fontsize=sz) 
        ax2[2,2].set_xlabel('Epoch', fontsize=sz)   
    
        ax2[0,0].set_ylabel('Loss', fontsize=sz) 
        ax2[1,0].set_ylabel('Loss', fontsize=sz) 
        ax2[2,0].set_ylabel('Loss', fontsize=sz) 
        
        #----------------------------------------------------------------------
        ax3.set_title('Mean Parameter Variation', fontsize=sz)
        ax3.set_ylabel('Cosine Distance', fontsize=sz) 
        ax3.set_xlabel('Epoch', fontsize=sz) 
        ax3.tick_params(axis='both', labelsize=sz)
        ax3.legend(legn,ncol=2,fontsize=sz-10,framealpha=0.5)
        ax3.grid()
        ax3.set_xlim([0, 100])
        
        #----------------------------------------------------------------------
        ax4.set_title('Mean Gradient Variation', fontsize=sz)
        ax4.set_ylabel('Modulus', fontsize=sz) 
        ax4.set_xlabel('Epoch', fontsize=sz) 
        ax4.tick_params(axis='both', labelsize=sz)
        ax4.legend(legn,ncol=2,fontsize=sz-10,framealpha=0.5)
        ax4.grid()
        ax4.set_xlim([0, 100])
        
        #----------------------------------------------------------------------   
            
        PT_tot.append(PT)



        
       

