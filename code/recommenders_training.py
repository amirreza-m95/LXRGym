# # Imports

import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import optuna
import logging
import matplotlib.pyplot as plt
import ipynb
import importlib

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single" #changed
}

num_users_dict = {
    "ML1M":6037,
    "Yahoo":13797, 
    "Pinterest":19155
}

num_items_dict = {
    "ML1M":3381,
    "Yahoo":4604, 
    "Pinterest":9362
}


data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "VAE" ## Can be MLP, VAE, MLP_model, GMF_model, NCF


DP_DIR = Path("processed_data", data_name) 
export_dir = Path(os.getcwd())
files_path = Path(export_dir, DP_DIR) #changes
checkpoints_path = Path(export_dir, "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

# ## Data imports and preprocessing

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

for row in range(static_test_data.shape[0]):
    static_test_data.iloc[row, static_test_data.iloc[row,-2]]=0
test_array = static_test_data.iloc[:,:-2].to_numpy()

static_test_data

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

# # Recommenders import

# from ipynb.fs.defs.recommenders_architecture import *
from recommenders_architecture import *
# importlib.reload(ipynb.fs.defs.recommenders_architecture)
# from ipynb.fs.defs.recommenders_architecture import *

# # Help functions

kw_dict = {'device':device,
          'num_items': num_items,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name}

# # Training

# ## MLP Train function

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

def MLP_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0, 4) # hyperparameter that weights the different loss terms
    epochs = 10
    model = MLP(hidden_dim, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    
    print(f'======================== new run - {recommender_name} ========================')
    logger.info(f'======================== new run - {recommender_name} ========================')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss=[]
        train_neg_loss=[]
        if epoch!=0 and epoch%10 == 0: # decrease the learning rate every 10 epochs
            lr = 0.1*lr
            optimizer.lr = lr
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device)

            batch_pos_idx = train_matrix[batch_idx,-2]
            batch_neg_idx = train_matrix[batch_idx,-1]
            
            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)
            
            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
            
            # MSE loss
            pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
            neg_loss = torch.mean((neg_output)**2)
            
            batch_loss = pos_loss + beta*neg_loss
            batch_loss.backward()
            optimizer.step()
            
            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())
            
        print(f'train pos_loss = {np.mean(train_pos_loss)}, neg_loss = {np.mean(train_neg_loss)}')    
        train_losses.append(np.mean(loss))
        torch.save(model.state_dict(), Path(checkpoints_path, f'MLP_{data_name}_{round(lr,4)}_{batch_size}_{trial.number}_{epoch}.pt'))


        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        
        test_pos = test_matrix[:,-2]
        test_neg = test_matrix[:,-1]
        
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        
        pos_items = torch.Tensor(items_array[test_pos]).to(device)
        neg_items = torch.Tensor(items_array[test_neg]).to(device)
        
        pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
        neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
        
        pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
        neg_loss = torch.mean((neg_output)**2)
        print(f'test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model,batch_index=batch_idx, **kw_dict)
        hr10.append(hit_rate_at_10) # metric for monitoring
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        test_losses.append(-hit_rate_at_10)
        if epoch>5: # early stop if the HR@10 decreases for 4 epochs in a row
            if test_losses[-2]<=test_losses[-1] and test_losses[-3]<=test_losses[-2] and test_losses[-4]<=test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
            
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

# ## VAE Train function

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

VAE_config= {
"enc_dims": [256,64],
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

def VAE_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [64,128,256])
    epochs = 20
    model = VAE(VAE_config ,**kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    print('======================== new run ========================')
    logger.info('======================== new run ========================')
    
    for epoch in range(epochs):
        if epoch!=0 and epoch%10 == 0:
            lr = 0.1*lr
            optimizer.lr = lr
        loss = model.train_one_epoch(train_array, optimizer, batch_size)
        train_losses.append(loss)
        torch.save(model.state_dict(), Path(checkpoints_path, f'VAE_{data_name}_{trial.number}_{epoch}_{round(lr,4)}_{batch_size}.pt'))


        model.eval()
        test_matrix = static_test_data.to_numpy()
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        test_pos = test_array[:,-2]
        test_neg = test_array[:,-1]
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        output = model(test_tensor).to(device)
        pos_loss = -output[row_indices,test_pos].mean()
        neg_loss = output[row_indices,test_neg].mean()
        print(f'pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        test_losses.append(pos_loss.item())
        if epoch>5:
            if test_losses[-2]<test_losses[-1] and test_losses[-3]<test_losses[-2] and test_losses[-4]<test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
    
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

# # lightGCN function

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

#### creating user item matrix -----------------
# create user-item matrix
# Load the ratings data
ratings = pd.read_csv('/home/amir/Documents/code/xrs/LXR/processed_data/ML1M/data files/ratings.dat', 
                      sep='::', header=None, engine='python', 
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

from scipy.sparse import coo_matrix

# Map user and item IDs to a contiguous range of indices
user_ids = ratings['UserID'].unique()
item_ids = ratings['MovieID'].unique()

user_map = {id: idx for idx, id in enumerate(user_ids)}
item_map = {id: idx for idx, id in enumerate(item_ids)}

ratings['UserIndex'] = ratings['UserID'].map(user_map)
ratings['ItemIndex'] = ratings['MovieID'].map(item_map)

# Create the user-item interaction matrix
user_item_matrix = coo_matrix((ratings['Rating'], (ratings['UserIndex'], ratings['ItemIndex'])), 
                              shape=(len(user_ids), len(item_ids)))
#----------------------------

print(user_item_matrix.shape)

def LightGCN_objective(trial):
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    epochs = 20
    alpha = trial.suggest_float('alpha', 0.01, 0.1)  # Degree of propagation
    
    kw_dict['num_users'] = num_users
    kw_dict['num_items'] = num_items
    kw_dict['device'] = device
    kw_dict['n_layers'] = n_layers
    kw_dict['alpha'] = alpha
    kw_dict['user_item_matrix'] = user_item_matrix  # Assuming prepared
    
    model = LightGCN(hidden_dim, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    
    print('======================== new run - LightGCN ========================')
    logger.info('======================== new run - LightGCN ========================')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))
    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        
        if epoch != 0 and epoch % 10 == 0:  # Decrease the learning rate every 10 epochs
            lr = 0.1 * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            # batch_matrix = torch.FloatTensor(train_matrix[batch_idx, :-2]).to(device)
            batch_matrix = torch.LongTensor(train_matrix[batch_idx, :-2]).to(device)
           
            batch_pos_idx = train_matrix[batch_idx, -2]
            batch_neg_idx = train_matrix[batch_idx, -1]
            
            pos_items = torch.LongTensor(batch_pos_idx).to(device)
            neg_items = torch.LongTensor(batch_neg_idx).to(device)
            # pos_items = torch.LongTensor(items_array[batch_pos_idx]).to(device)
            # neg_items = torch.LongTensor(items_array[batch_neg_idx]).to(device)
            
            pos_output = torch.diagonal(model(batch_matrix, pos_items))
            neg_output = torch.diagonal(model(batch_matrix, neg_items))
            
            pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
            neg_loss = torch.mean(neg_output ** 2)
            
            batch_loss = pos_loss + neg_loss
            batch_loss.backward()
            optimizer.step()
            
            loss.append(batch_loss.item())
        
        print(f'Epoch {epoch}: train loss = {np.mean(loss)}')
        train_losses.append(np.mean(loss))
        
        torch.save(model.state_dict(), Path(checkpoints_path, f'LightGCN_{data_name}_{round(lr, 4)}_{batch_size}_{trial.number}_{epoch}.pt'))

        model.eval()
        test_loss = []
        all_hit_rates = []
        num_test = static_test_data.shape[0]
        num_test_batches = int(np.ceil(num_test / batch_size))
        perm = np.random.permutation(num_test)
        
        # Testing Loop in Batches
        for b in range(num_test_batches):
            if (b + 1) * batch_size >= num_test:
                # batch_idx = perm[b * batch_size, num_test]
                batch_idx = perm[b * batch_size:]
            else:
                # batch_idx = perm[b * batch_size, (b + 1) * batch_size] # do it with permutation
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            # test_matrix = static_test_data.iloc[batch_idx] #changes
            # test_tensor = torch.tensor(test_matrix[:-2].values).to(device) # changed
            
            # test_pos = test_matrix.iloc[:, -2]
            # test_neg = test_matrix.iloc[:, -1]
            
            # row_indices = np.arange(test_matrix.shape[0])
            # test_tensor[row_indices, test_pos] = 0
            test_matrix = np.array(static_test_data)[batch_idx]
            test_tensor = torch.LongTensor(test_matrix[:, :-2]).to(device)
            
            test_pos = test_matrix[:, -2]
            test_neg = test_matrix[:, -1]
            
            row_indices = np.arange(test_matrix.shape[0])
            test_tensor[row_indices, test_pos] = 0
        
            
            pos_items = torch.LongTensor(test_pos).to(device)
            neg_items = torch.LongTensor(test_neg).to(device)
            
            pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
            neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
            
            pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
            neg_loss = torch.mean(neg_output ** 2)
            
            test_loss.append(pos_loss.item())
            
            hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, batch_index=batch_idx ,**kw_dict)
            all_hit_rates.append(hit_rate_at_10)
        
        avg_test_loss = np.mean(test_loss)
        avg_hr10 = np.mean(all_hit_rates)
        
        print(f'Epoch {epoch}: test pos_loss = {avg_test_loss}, hr10 = {avg_hr10}')
        hr10.append(avg_hr10)
        test_losses.append(avg_test_loss)
        
        if epoch > 5:
            if test_losses[-2] < test_losses[-1] and test_losses[-3] < test_losses[-2] and test_losses[-4] < test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
    
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)
        # model.eval()
        # test_matrix = np.array(static_test_data)
        # test_tensor = torch.LongTensor(test_matrix[:, :-2]).to(device)
        
        # test_pos = test_matrix[:, -2]
        # test_neg = test_matrix[:, -1]
        
        # row_indices = np.arange(test_matrix.shape[0])
        # test_tensor[row_indices, test_pos] = 0
        
        # pos_items = torch.LongTensor(test_pos).to(device)
        # neg_items = torch.LongTensor(test_neg).to(device)
        
        # pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
        # neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
        
        # pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
        # neg_loss = torch.mean(neg_output ** 2)
        # print(f'Epoch {epoch}: test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        # hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        # hr10.append(hit_rate_at_10)
        # print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        # test_losses.append(pos_loss.item())
        # if epoch > 5:
        #     if test_losses[-2] < test_losses[-1] and test_losses[-3] < test_losses[-2] and test_losses[-4] < test_losses[-3]:
        #         logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
        #         train_losses_dict[trial.number] = train_losses
        #         test_losses_dict[trial.number] = test_losses
        #         HR10_dict[trial.number] = hr10
        #         return max(hr10)
        
        # logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
        # train_losses_dict[trial.number] = train_losses
        # test_losses_dict[trial.number] = test_losses
        # HR10_dict[trial.number] = hr10
        # return max(hr10)

# # Help Functions
# 

import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# a function that samples different train data variation for a diverse training
def sample_indices(data, **kw):
    num_items = kw['num_items']
    pop_array = kw['pop_array']
    
    matrix = np.array(data)[:,:num_items] # keep only items columns, remove demographic features columns
    zero_indices = []
    one_indices = []

    for row in matrix:
        zero_idx = np.where(row == 0)[0]
        one_idx = np.where(row == 1)[0]
        probs = pop_array[zero_idx]
        probs = probs/ np.sum(probs)

        sampled_zero = np.random.choice(zero_idx, p = probs) # sample negative interactions according to items popularity 
        zero_indices.append(sampled_zero)

        sampled_one = np.random.choice(one_idx) # sample positive interactions from user's history
        data.iloc[row, sampled_one] = 0
        one_indices.append(sampled_one)

    data['pos'] = one_indices
    data['neg'] = zero_indices
    return np.array(data)

# evaluate recommenders on test set and return HR@10, HR@50, HR@100, MRR and MPR

# def recommender_evaluations(recommender, **kw):
#     static_test_data = kw['static_test_data'].copy()
#     device = kw['device']
#     items_array = kw['items_array']
#     num_items = kw['num_items']
#     batch_size = kw.get('batch_size', 64)  # Default batch size if not provided
    
#     counter_10 = 0
#     counter_50 = 0
#     counter_100 = 0
#     RR = 0
#     PR = 0
    
#     temp_test_array = np.array(static_test_data)
#     num_test = temp_test_array.shape[0]
#     num_batches = int(np.ceil(num_test / batch_size))
    
#     for b in range(num_batches):
#         if (b + 1) * batch_size >= num_test:
#             batch_idx = np.arange(b * batch_size, num_test)
#         else:
#             batch_idx = np.arange(b * batch_size, (b + 1) * batch_size)
        
#         batch_test_data = temp_test_array[batch_idx]
#         batch_users = torch.Tensor(batch_test_data[:, :-2]).to(device)
#         batch_items_pos = batch_test_data[:, -2]
        
#         # Mask the positive items in the user vectors
#         row_indices = np.arange(batch_users.shape[0])
#         batch_users[row_indices, batch_items_pos] = 0
        
#         for i, user_tensor in enumerate(batch_users):
#             item_id = batch_items_pos[i]
#             item_tensor = items_array[item_id]
            
#             index = get_index_in_the_list(user_tensor, user_tensor, item_id, recommender, **kw) + 1
#             if index <= 10:
#                 counter_10 += 1
#             if index <= 50:
#                 counter_50 += 1
#             if index <= 100:
#                 counter_100 += 1
#             RR += np.reciprocal(index)
#             PR += index / num_items
    
#     n = temp_test_array.shape[0]
#     return counter_10 / n, counter_50 / n, counter_100 / n, RR / n, PR * 100 / n

def recommender_evaluations(recommender, **kw):
    static_test_data = kw['static_test_data'].copy()#.iloc[batch_index]
    device = kw['device']
    items_array = kw['items_array']
    num_items = kw['num_items']

    counter_10 = 0
    counter_50 = 0
    counter_100 = 0
    RR = 0
    PR = 0
    temp_test_array = np.array(static_test_data)
    n = temp_test_array.shape[0]
    for i in range(n):
        item_id = temp_test_array[i][-2]
        item_tensor = items_array[item_id]
        user_tensor = torch.Tensor(temp_test_array[i][:-2]).to(device)
        user_tensor[item_id]=0
        index = get_index_in_the_list(user_tensor, user_tensor, item_id, recommender, **kw) +1 
        if index <= 10:
            counter_10 +=1 
        if index <= 50:
            counter_50 +=1 
        if index <= 100:
            counter_100 +=1             
        RR += np.reciprocal(index)
        PR += index/num_items
        
    return counter_10/n, counter_50/n, counter_100/n,  RR/n, PR*100/n

# a function that returns a specific item's rank in user's recommendations list
def get_index_in_the_list(user_tensor, original_user_tensor, item_id, recommender, **kw):
    top_k_list = list(get_top_k(user_tensor, original_user_tensor, recommender, **kw).keys())
    return top_k_list.index(item_id)

# returns a dictionary of items and recommendations scores for a specific user
def get_top_k(user_tensor, original_user_tensor, model, **kw):
    # all_items_tensor = kw['all_items_tensor']
    # num_items = kw['num_items']
    # batch_size = kw.get('batch_size', 64)
    
    # item_prob_dict = {}
    # original_user_vector = np.array(original_user_tensor.cpu())[:num_items]
    # catalog = np.ones_like(original_user_vector) - original_user_vector
    
    # # Process items in batches
    # for i in range(0, num_items, batch_size):
    #     batch_items_tensor = all_items_tensor[i:i + batch_size].to(user_tensor.device)
    #     output_model = recommender_run(user_tensor, model, batch_items_tensor, None, 'vector', **kw)
    #     output_model = output_model.cpu().detach().numpy()
        
    #     # Ensure catalog and output_model have compatible shapes for multiplication
    #     catalog_batch = catalog[i:i + batch_size]
    #     output_model = catalog_batch * output_model
        
    #     for j in range(len(output_model)):
    #         if catalog_batch[j] > 0:
    #             item_prob_dict[i + j] = output_model[j]
    
    # sorted_items_by_prob = sorted(item_prob_dict.items(), key=lambda item: item[1], reverse=True)
    # return dict(sorted_items_by_prob)
    all_items_tensor = kw['all_items_tensor']
    num_items = kw['num_items']
    
    item_prob_dict = {}
    output_model = [float(i) for i in recommender_run(user_tensor, model, all_items_tensor, None, 'vector', **kw).cpu().detach().numpy()]
    original_user_vector = np.array(original_user_tensor.cpu())[:num_items]
    catalog = np.ones_like(original_user_vector)- original_user_vector
    output = catalog*output_model
    for i in range(len(output)):
        if catalog[i] > 0:
            item_prob_dict[i]=output[i]
    sorted_items_by_prob  = sorted(item_prob_dict.items(), key=lambda item: item[1],reverse=True)
    return dict(sorted_items_by_prob)

# a function that wraps the different recommenders types 
# returns user's scores with respect to a certain item or for all items 
def recommender_run(user_tensor, recommender, item_tensor = None, item_id= None, wanted_output = 'single', **kw):
    output_type=kw['output_type']
    if output_type == 'single':
        if wanted_output == 'single':
            return recommender(user_tensor, item_tensor)
        else:
            return recommender(user_tensor, item_tensor).squeeze()
    else:
        if wanted_output == 'single':
            return recommender(user_tensor).squeeze()[item_id]
        else:
            return recommender(user_tensor).squeeze()
# def recommender_run(user_tensor, recommender, item_tensor=None, item_id=None, wanted_output='single', **kw):
#     output_type = kw['output_type']
#     if output_type == 'single':
#         if wanted_output == 'single':
#             return recommender(user_tensor, item_tensor)
#         else:
#             return recommender(user_tensor, item_tensor).squeeze()
#     else:
        if wanted_output == 'single':
            return recommender(user_tensor).squeeze()[item_id]
        else:
            return recommender(user_tensor).squeeze()
# # save logs

# ### Save logs in text file, optimize using Optuna

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(f"{recommender_name}_{data_name}_Optuna.log", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

study = optuna.create_study(direction='maximize')

logger.info("Start optimization.")

if recommender_name == 'MLP':
    study.optimize(MLP_objective, n_trials=20) 
elif recommender_name == 'VAE':
    study.optimize(VAE_objective, n_trials=20) 
elif recommender_name == 'LightGCN':
    study.optimize(LightGCN_objective, n_trials=5)

with open(f"{recommender_name}_{data_name}_Optuna.log") as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"
    
    
# Print best hyperparameters and corresponding metric value
print("Best hyperparameters: {}".format(study.best_params))
print("Best metric value: {}".format(study.best_value))

for run in HR10_dict.keys():
    print(run, np.argmax(HR10_dict[run]), max(HR10_dict[run]))
    plt.plot(HR10_dict[run])
plt.legend(HR10_dict.keys(), loc='upper left')
plt.show()

# # Evaluations
# ### Load the trained recommender

from ipynb.fs.defs.help_functions import *
importlib.reload(ipynb.fs.defs.help_functions)
from ipynb.fs.defs.help_functions import *

recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),

    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
}

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]

def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)
    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path), map_location=torch.device('cpu'))
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    return recommender
    
model = load_recommender()

# ### Plot the distribution of top recommended item accross all users

# plot the distribution of top recommended item accross all users
topk_train = {}
for i in range(len(train_array)):
    vec = train_array[i]
    tens = torch.Tensor(vec).to(device)
    topk_train[i] = int(get_user_recommended_item(tens, model, **kw_dict).cpu().detach().numpy())

plt.hist(topk_train.values(), bins=1000)
plt.plot(np.array(list(pop_dict.keys())), np.array(list(pop_dict.values()))*100, alpha=0.2)
plt.show()

topk_test = {}
for i in range(len(test_array)):
    vec = test_array[i]
    tens = torch.Tensor(vec).to(device)
    topk_test[i] = int(get_user_recommended_item(tens, model, **kw_dict).cpu().detach().numpy())

plt.hist(topk_test.values(), bins=400)
plt.plot(np.array(list(pop_dict.keys())), np.array(list(pop_dict.values()))*50, alpha=0.2)
plt.show() 

# ### Calculate recommender's HR@10, HR@50, HR@100, MRR and MPR

hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)

print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)