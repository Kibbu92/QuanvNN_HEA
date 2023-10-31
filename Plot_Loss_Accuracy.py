# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:18:41 2023

@author: andca
"""

import os
import numpy as np
import matplotlib.pyplot as plt

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
    
    ind = 0
    sz = 15 
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    folder = os.getcwd() + '\\results\\MNIST' 
 
    Layer_Krn = os.listdir(folder)
    for i in range(len(Layer_Krn)):
        print('')
        legn = []    

        fig0,ax0 = plt.subplots(nrows = 3, ncols = 3)
        fig1,ax1 = plt.subplots(nrows = 3, ncols = 3)
        fig2,ax2 = plt.subplots(nrows = 3, ncols = 3)

        fig3,ax3 = plt.subplots(nrows = 2, ncols = 1)

        Folder_i = folder+ '\\' + Layer_Krn[i]
        OPT = os.listdir(Folder_i)  
        
        Acc_train_opt = []
        Acc_test_opt  = []
        for k in range(len(OPT)):
        
            Folder_opt = Folder_i + '\\' + OPT[k] 
            
            File  = os.listdir(Folder_opt)
            
            len_m = len(File)//2
            
            
            Acc_train = []
            Acc_test  = []   
            Loss = []

            Parm = []
            Grad = []
            for m in range(len_m):
                
                with open(Folder_opt + '\\' + File[m],'rb') as f:
                    acc_train_traj = np.load(f,allow_pickle=True)   
                    acc_test_traj  = np.load(f,allow_pickle=True)
                
                with open(Folder_opt + '\\' + File[len_m+m],'rb') as f:
                    cost_params = np.load(f,allow_pickle=True) 
                    param_traj  = np.load(f,allow_pickle=True) 
                    loss_traj   = np.load(f,allow_pickle=True)                  
                 
                
                loss_traj = loss_traj.reshape(len(loss_traj)//20,20)[:,-1]   
                pt = flat_traj(param_traj)
                pt = pt[:-1,:]
                pt = pt.reshape(len(pt)//20,20,-1)[:,-1,:]  
   
                text = f'{OPT[k].replace(Layer_Krn[i],"")[0:-1]}'     
                #==============================================================
                ax0[k//3,k-(k//3)*3].plot(loss_traj) 
                ax0[k//3,k-(k//3)*3].tick_params(axis='both', labelsize=sz)
                ax0[k//3,k-(k//3)*3].text(0.5, 0.85, text, transform=ax0[k//3,k-(k//3)*3].transAxes, fontsize=sz,
                        verticalalignment='top', bbox=props)
                ax0[k//3,k-(k//3)*3].grid()
                ax0[k//3,k-(k//3)*3].set_xlim([0, 95])
                ax0[k//3,k-(k//3)*3].set_ylim([0, 60])
                
                #==============================================================
                Epoch_acc = np.arange(0,100,5)

                ax1[k//3,k-(k//3)*3].plot(Epoch_acc,acc_train_traj) 
                ax1[k//3,k-(k//3)*3].tick_params(axis='both', labelsize=sz)
                ax1[k//3,k-(k//3)*3].text(0.5, 0.3, text, transform=ax1[k//3,k-(k//3)*3].transAxes, fontsize=sz,
                        verticalalignment='top', bbox=props)
                ax1[k//3,k-(k//3)*3].grid()
                ax1[k//3,k-(k//3)*3].set_xlim([0, 95])
                ax1[k//3,k-(k//3)*3].set_ylim([0, 1.1])

                #==============================================================                
                
                ax2[k//3,k-(k//3)*3].plot(Epoch_acc,acc_test_traj) 
                ax2[k//3,k-(k//3)*3].tick_params(axis='both', labelsize=sz)
                ax2[k//3,k-(k//3)*3].text(0.5, 0.3, text, transform=ax2[k//3,k-(k//3)*3].transAxes, fontsize=sz,
                        verticalalignment='top', bbox=props)
                ax2[k//3,k-(k//3)*3].grid()
                ax2[k//3,k-(k//3)*3].set_xlim([0, 95])
                ax2[k//3,k-(k//3)*3].set_ylim([0, 1.1])

                #==============================================================                
                Acc_train.append(acc_train_traj)
                Acc_test.append(acc_test_traj)  
                Loss.append(loss_traj)
                Parm.append(pt)
                
            #==================================================================
            #==================================================================

            Acc_Train_mean = np.mean(Acc_train,axis=0)
            Acc_Test_mean = np.mean(Acc_test,axis=0)   
            Loss_mean = np.mean(Loss,axis=0)
            Parm_mean = np.mean(Parm,axis=0)
            
            ind = np.where(Acc_Train_mean==1)[0]
            if len(ind)==0:
                ind = 10000
            else:
                ind = ind[0]
                
            print(ind)

            ax3[0].plot(Epoch_acc,Acc_Train_mean)
            ax3[0].grid()
            ax3[0].set_xlim([0, 95])
            ax3[0].set_ylim([0, 1.1])
            
            ax3[1].plot(Loss_mean)
            ax3[1].grid()
            ax3[1].set_xlim([0, 95])
            ax3[1].set_ylim([0, 60])
            
            legn.append(  OPT[k].replace(Layer_Krn[i],"")[0:-1] )    
            
       
        #======================================================================
        #======================================================================
        ax0[0,1].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        
        ax0[2,0].set_xlabel('Epoch', fontsize=sz) 
        ax0[2,1].set_xlabel('Epoch', fontsize=sz) 
        ax0[2,2].set_xlabel('Epoch', fontsize=sz)   
        
        ax0[0,0].set_ylabel('Loss', fontsize=sz) 
        ax0[1,0].set_ylabel('Loss', fontsize=sz) 
        ax0[2,0].set_ylabel('Loss', fontsize=sz) 
            
        fig0.tight_layout()
        fig0.savefig(os.path.join(os.getcwd() + '\\results\\', 'Loss'+f'_{Layer_Krn[i]}'+'.png'))
        #======================================================================
        ax1[0,1].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        
        ax1[2,0].set_xlabel('Epoch', fontsize=sz) 
        ax1[2,1].set_xlabel('Epoch', fontsize=sz) 
        ax1[2,2].set_xlabel('Epoch', fontsize=sz)   
        
        ax1[0,0].set_ylabel('Train Accuracy', fontsize=sz) 
        ax1[1,0].set_ylabel('Train Accuracy', fontsize=sz) 
        ax1[2,0].set_ylabel('Train Accuracy', fontsize=sz) 
            
        fig1.tight_layout()
        fig1.savefig(os.path.join(os.getcwd() + '\\results\\', 'Train_acc'+f'_{Layer_Krn[i]}'+'.png'))
             
        
        #======================================================================
        ax2[0,1].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)

        ax2[2,0].set_xlabel('Epoch', fontsize=sz) 
        ax2[2,1].set_xlabel('Epoch', fontsize=sz) 
        ax2[2,2].set_xlabel('Epoch', fontsize=sz)   
        
        ax2[0,0].set_ylabel('Test Accuracy', fontsize=sz) 
        ax2[1,0].set_ylabel('Test Accuracy', fontsize=sz) 
        ax2[2,0].set_ylabel('Test Accuracy', fontsize=sz)  
        
        
        fig2.tight_layout()
        fig2.savefig(os.path.join(os.getcwd() + '\\results\\', 'Test_acc'+f'_{Layer_Krn[i]}'+'.png'))
        
        #======================================================================
        ax3[0].set_title(f'Layer_Kernel = {Layer_Krn[i]}', fontsize=sz)
        ax3[0].set_ylabel('Mean Train Accuracy', fontsize=sz) 
        ax3[0].tick_params(axis='both', labelsize=sz)
        ax3[0].legend(legn)
        
        ax3[1].set_xlabel('Epoch', fontsize=sz) 
        ax3[1].set_ylabel('Mean Loss Function', fontsize=sz) 
        fig3.tight_layout()
        fig3.savefig(os.path.join(os.getcwd() + '\\results\\', 'Mean_Acc_Loss'+f'_{Layer_Krn[i]}'+'.png'))
        
       

