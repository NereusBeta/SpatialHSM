
import torch
from .preprocess import *
#from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import *
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import normalize

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import scipy.sparse as sp
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_


import math
import torch
from torch.optim.optimizer import Optimizer

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import psutil
import torch

import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_memory_usage():
    """Get current RAM usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_gpu_memory():
    """Get current GPU memory usage in MB (if available)"""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved": torch.cuda.memory_reserved() / (1024 * 1024)
        }
    return {"allocated": 0, "reserved": 0}


def get_laplacian_mtx(X, num_neighbors=6, normalization=False):
    """
    Compute the Laplacian matrix for graph representation of the input data.
    
    Parameters:
    - X: input data (samples x features)
    - num_neighbors: number of nearest neighbors to consider for the graph
    - normalization: whether to normalize the Laplacian matrix
    
    Returns:
    - Laplacian matrix
    """
    from sklearn.neighbors import kneighbors_graph
    
    adj_mtx = kneighbors_graph(X, n_neighbors=num_neighbors, mode='connectivity', include_self=True)
    adj_mtx = ss.csr_matrix(adj_mtx)

    deg_mtx = np.array(adj_mtx.sum(axis=1)).flatten()
    deg_mtx = np.diagflat(deg_mtx)

    if not normalization:
        lap_mtx = deg_mtx - adj_mtx
    else:
        deg_inv_sqrt = np.diagflat(np.power(deg_mtx.diagonal(), -0.5))
        lap_mtx = np.identity(deg_mtx.shape[0]) - deg_inv_sqrt @ adj_mtx @ deg_inv_sqrt
    return lap_mtx

def obtain_freq_spots(adata, lap_mtx, n_fcs, c=1):
    """
    Extract frequency domain features from the data using the Laplacian matrix.
    
    Parameters:
    - adata: input data (samples x features)
    - lap_mtx: Laplacian matrix
    - n_fcs: number of frequency components to extract
    - c: scaling factor for frequency components
    
    Returns:
    - frequency domain feature matrix
    """
    if isinstance(adata, torch.Tensor):
        X = adata.cpu().detach().numpy()
    else:
        X = adata if not ss.issparse(adata) else adata.A

    X = normalize(X, norm='max', axis=0)
    X = np.matmul(X, X.T)
    X = normalize(X)

    n_fcs = min(n_fcs, X.shape[0] - 1)

    v0 = [1 / np.sqrt(lap_mtx.shape[0])] * lap_mtx.shape[0]
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx, k=n_fcs, which='SM', v0=v0)
    power = [1 / (1 + c * eigv) for eigv in eigvals]
    eigvecs = np.matmul(eigvecs, np.diag(power))

    freq_mtx = np.matmul(eigvecs.T, X)
    freq_mtx = normalize(freq_mtx[1:, :], norm='l2', axis=0).T
    return freq_mtx, eigvecs.T


def apply_fourier_transform(freq_features):
    """
    Perform Fourier transform on frequency domain features to extract magnitude.
    
    Parameters:
    - freq_features: frequency domain feature matrix
    
    Returns:
    - magnitude of the Fourier transform
    """
    fft_transformed = np.fft.fft(freq_features, axis=1)  
    magnitude = np.abs(fft_transformed)  
    return magnitude

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor


def filter_frequency_signal(freq_signal, method='wavelet', smooth=True, sigma=1, wavelet='coif4', level=3):
    """
    Filter frequency domain signals and support wavelet transform and Gaussian smoothing.
    
Parameters:
-Freq_Signal: Input frequency signal
-Method: Filtering method (only supports' wavelet ')
-Smooth: Enable Gaussian smoothing or not
-Sigma: Gaussian smoothing parameter
-Wavelet: Wavelet basis functions (such as'db1 ','sym4', 'coif4', etc.)
-Level: Number of wavelet decomposition layers

return:
-Filtered signal: filtered signal
    """
    
    if method == 'wavelet':
        coeffs = pywt.wavedec(freq_signal, wavelet, level=level)

        
        threshold_value = np.sqrt(2 * np.log(freq_signal.size)) * 0.5
      
        with ThreadPoolExecutor() as executor:
            coeffs[1:] = list(executor.map(lambda c: pywt.threshold(c, value=threshold_value, mode='soft'), coeffs[1:]))

        # wavelet reconstruction
        filtered_signal = pywt.waverec(coeffs, wavelet)[:len(freq_signal)]
    else:
        raise ValueError("Unsupported method. Only 'wavelet' is supported.")

    
    if smooth:
        
        filtered_signal = gaussian_filter1d(filtered_signal, sigma=sigma)
    
    return filtered_signal


def plot_signals_overall(original_signals, filtered_signals, title="Overall Signal Comparison"):
    """
    Draw the overall signal and average the signals of all dimensions before drawing.
    
Parameters:
-Origina_signals: raw signal matrix (n_samples, n_features)
-Filtered_stignals: The filtered signal matrix (n_samples, n_features)
-Title: Chart Title
    """
  
    avg_original_signal = np.mean(original_signals, axis=1)
    avg_filtered_signal = np.mean(filtered_signals, axis=1)

 
    plt.figure(figsize=(12, 6))
    plt.plot(avg_original_signal, label="Average Original Signal", alpha=0.7, color="blue")
    plt.plot(avg_filtered_signal, label="Average Filtered Signal", alpha=0.7, color="orange")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def augment_data(features, noise_level=0.01):
    """
    Add Gaussian noise for data augmentation.

Parameters:
-Features: Input features
-Noise_level: noise intensity

return:
-Enhanced features
    """
    noise = torch.randn_like(features) * noise_level
    return features + noise


class SpatialHSM():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        #learning_rate=0.0005,
        learning_rate=2e-4,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=1000, 
        dim_input=3000,
        dim_output=64,
        
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X'
        ):
        """
        Initialize the SpatialHSM model with parameters for spatial transcriptomics analysis.

        Args:
            adata (AnnData): The input AnnData object containing spatial transcriptomics data.
            adata_sc (AnnData, optional): Single-cell RNA-seq data. Default is None.
            device (torch.device, optional): The device to run the model on (CPU or GPU). Default is CPU.
            learning_rate (float, optional): Learning rate for model training. Default is 2e-4.
            learning_rate_sc (float, optional): Learning rate for single-cell model training. Default is 0.01.
            weight_decay (float, optional): Weight decay (L2 regularization). Default is 0.00.
            epochs (int, optional): Number of training epochs. Default is 1000.
            dim_input (int, optional): Input feature dimension. Default is 3000.
            dim_output (int, optional): Output feature dimension. Default is 64.
            random_seed (int, optional): Random seed for reproducibility. Default is 41.
            alpha (float, optional): Weight for feature loss in the final loss calculation. Default is 10.
            beta (float, optional): Weight for CSL (contrastive learning) loss. Default is 1.
            theta (float, optional): Hyperparameter used for controlling loss. Default is 0.1.
            lamda1 (float, optional): Regularization parameter for L1 loss. Default is 10.
            lamda2 (float, optional): Regularization parameter for Noise Contrastive Estimation (NCE). Default is 1.
            deconvolution (bool, optional): Whether to perform deconvolution for single-cell RNA-seq data. Default is False.
            datatype (str, optional): Data type (e.g., '10X', 'Stereo', 'Slide'). Default is '10X'.
        """
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)
           
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           print ('Extracting feature ...') 
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
        # Extract original features
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        # print("SGFN -Freq")
        self.integrate_frequency_features()  # Frequency domain feature fusion
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
    def integrate_frequency_features(self):
        """
        Calculate frequency domain features using Laplacian matrix and integrate them with the original feature matrix.
        Apply PCA to reduce the combined features' dimensionality.

        The process involves the following steps:
        1. Compute the Laplacian matrix.
        2. Extract frequency domain features.
        3. Filter frequency signals using wavelet transformation.
        4. Concatenate original features with the filtered frequency features.
        5. Apply PCA for dimensionality reduction.

        This method modifies the 'self.features' attribute in place by reducing its dimensions to match the original size.
        """
        # Start monitoring
        start_time = time.time()
        start_mem = get_memory_usage()
        
        print("Integrating frequency features...")

        # Step 1: Laplacian matrix
        lap_mtx = get_laplacian_mtx(self.features.cpu().numpy(), num_neighbors=6, normalization=False)

        # Step 2: Extract frequency features
        freq_features, _ = obtain_freq_spots(adata=self.features, lap_mtx=lap_mtx, n_fcs=10, c=2)

        # Step 3: Filter frequency signals
        filtered_freq_features = np.array([
            filter_frequency_signal(freq_signal, method='wavelet', smooth=True, sigma=1) 
            for freq_signal in freq_features.T
        ]).T
        
        # Visualize signals
        plot_signals_overall(freq_features, filtered_freq_features, title="Frequency Signal Comparison")
        
        # Step 4: Feature concatenation
        freq_features_tensor = torch.tensor(filtered_freq_features, dtype=torch.float32).to(self.device)
        combined_features = torch.cat((self.features, freq_features_tensor), dim=1)

        # Step 5: PCA dimension reduction to original feature dimensions
        print("Performing PCA to reduce dimensions...")
        combined_features_np = combined_features.cpu().detach().numpy()
        pca = PCA(n_components=self.features.shape[1])
        reduced_features = pca.fit_transform(combined_features_np)

        # Convert back to tensor and save
        self.features = torch.tensor(reduced_features, dtype=torch.float32).to(self.device)
        print(f"Features after PCA reduction: {self.features.shape}")
        
        # End monitoring and report
        end_time = time.time()
        end_mem = get_memory_usage()
        
        print(f"Frequency feature integration completed in {end_time - start_time:.2f} seconds")
        print(f"Memory used: {end_mem - start_mem:.2f} MB")
        if torch.cuda.is_available():
            gpu_mem = get_gpu_memory()
            print(f"GPU memory: {gpu_mem['allocated']:.2f} MB allocated, {gpu_mem['reserved']:.2f} MB reserved")
    
    def plot_loss(self, loss_history):
        """
        Plots the training loss over epochs.

        Args:
            loss_history (list): A list of loss values recorded during training.

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(loss_history)), loss_history, label='Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def train(self):
        """
        Train the model using spatial transcriptomics data.

        This method trains the SpatialHSM model, performs backpropagation, and includes early stopping if the loss does not improve.
        
        Returns:
            AnnData: The input AnnData object with updated embeddings.
        """
        start_time = time.time()
        start_mem = get_memory_usage()
      
        if self.datatype in ['Stereo', 'Slide']:
                self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = EncoderAttentionResidualMLP(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
      
        
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5 
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6  
        )
        
        # Define gradient clipping range and early stopping mechanism
        max_grad_norm = 1.0  
        patience = 30 # Early Stopping
        best_loss = float('inf')
        early_stop_count = 0

        print("Begin to train ST data ...")
        self.model.train()
        loss_history = []

        preprocessed_features_a = F.normalize(permutation(self.features), p=2, dim=1).to(self.device)

        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            
            self.features_a = preprocessed_features_a

          
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)

         
            loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            loss_feat = F.mse_loss(self.features, self.emb)
            
            loss = self.alpha * loss_feat + self.beta * (loss_sl_1 + loss_sl_2)

         
            l1_lambda = 1e-6 
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())  # L1 范数计算
            loss += l1_lambda * l1_norm

            loss_history.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

          
            self.optimizer.step()
            scheduler.step()

            # Early Stopping 
            if loss.item() < best_loss:
                best_loss = loss.item()  
                early_stop_count = 0  
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print(f"Early stopping at epoch {epoch} with loss {loss.item():.4f}")
                    break  

        print("Optimization finished for ST data!")


        self.plot_loss(loss_history)
        # End monitoring and report
        end_time = time.time()
        end_mem = get_memory_usage()
        
        print("\n====== TRAINING PERFORMANCE SUMMARY ======")
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        print(f"Memory used: {end_mem - start_mem:.2f} MB")
        if torch.cuda.is_available():
            gpu_mem = get_gpu_memory()
            print(f"GPU memory: {gpu_mem['allocated']:.2f} MB allocated, {gpu_mem['reserved']:.2f} MB reserved")
        print("==========================================\n")

        with torch.no_grad():
            self.model.eval()
            if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                return self.emb_rec
            else:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                return self.adata 
    

    
    import torch
    from sklearn.decomposition import PCA
    import numpy as np

    def preprocess_features(self):
        
        if isinstance(self.features, torch.Tensor):
            features_np = self.features.cpu().detach().numpy()
        else:
            features_np = self.features
        
        pca = PCA(n_components=338)
        features_reduced = pca.fit_transform(features_np)

        self.features = torch.FloatTensor(features_reduced).to(self.device)
        
        print(f"特征维度已从 {features_np.shape} 调整为 {self.features.shape}")

        if hasattr(self, 'features_a') and self.features_a is not None:
            if isinstance(self.features_a, torch.Tensor):
                features_a_np = self.features_a.cpu().detach().numpy()
            else:
                features_a_np = self.features_a

            features_a_reduced = pca.transform(features_a_np) 
            self.features_a = torch.FloatTensor(features_a_reduced).to(self.device)
            
            print(f"features_a维度已从 {features_a_np.shape} 调整为 {self.features_a.shape}")

    
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        
        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
          
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
             
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
            
        print("Mapping matrix learning finished!")
        
        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc


    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE

    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
    
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss

    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M        
