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
from prettytable import PrettyTable


data_name = "ML1M" ### Can be ML1M, ML1M_demographic, Yahoo, Pinterest
recommender_name = "MLP" ### Can be MLP, VAE, MLP_model, GMF_model, NCF, LightGCN

DP_DIR = Path("processed_data", data_name) 
# export_dir = Path(os.getcwd()) #changed 

current_directory = os.getcwd()
path_parts = current_directory.split(os.sep)
# export_dir = os.sep.join(path_parts[:-1])
export_dir = os.sep.join(path_parts[:])

files_path = Path(export_dir, DP_DIR)
min_num_of_items_per_user = 2
min_num_of_users_per_item = 2
checkpoints_path = Path(export_dir, "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

movies = pd.read_csv(
    '/home/amir/Documents/code/xrs/LXR/processed_data/ML1M/data files/movies.dat',
    engine='python',
    delimiter='::',
    names=['movieid', 'title', 'genre'],
    index_col='movieid',
    encoding='latin1' #fuckin hell
)


# movies = pd.read_csv('/home/amir/Documents/code/xrs/LXR/processed_data/ML1M/data files/movies.csv', engine='python',
#                          delimiter=',', names=['movieid', 'title', 'genre']).set_index('movieid')

# movies


# movies = pd.read_csv(Path(files_path,'data files/movies.csv'), index_col=0)
# movies = pd.read_csv(Path(files_path,'data files/movies.dat'),delimiter='::',header=None,
#         names=['user_id', 'movie_id', 'rating', 'timestamp'], 
#         usecols=['user_id', 'movie_id', 'rating'], engine='python')

output_type_dict = { #changed
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single", #changed
    "NCF": "single",
    "MLP_model": "single",
    "GMF_model": "single"
}

num_users_dict = {
    "ML1M":6037,
    "ML1M_demographic":6037,
    "Yahoo":13797, 
    "Pinterest":19155
}

num_items_dict = {
    "ML1M":3381,
    "ML1M_demographic":3381,
    "Yahoo":4604, 
    "Pinterest":9362
}

demographic_dict = {
    "ML1M_demographic": True,
    "ML1M":False,
    "Yahoo":False, 
    "Pinterest":False
}

features_dict = {
    "ML1M_demographic": 3421,
    "ML1M":None,
    "Yahoo":None, 
    "Pinterest":None
}

recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","LightGCN"): Path(checkpoints_path, "LightGCN_ML1M_0.0001_64_3.pt"),#changed
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),
    ("ML1M","MLP_model"):Path(checkpoints_path, "MLP_model_ML1M_0.0001_64_27.pt"),
    ("ML1M","GMF_model"): Path(checkpoints_path, "GMF_best_ML1M_0.0001_32_17.pt"),
    ("ML1M","NCF"):Path(checkpoints_path, "NCF_ML1M_5e-05_64_16.pt"),

    ("ML1M_demographic","VAE"): Path(checkpoints_path, "VAE_ML1M_demographic_0.0001_64_6_18.pt"),
    ("ML1M_demographic","MLP"):Path(checkpoints_path, "MLP_ML1M_demographic_0.0_64_0_28.pt"),
    ("ML1M_demographic","MLP_model"):Path(checkpoints_path, "MLP_model2_ML1M_demographic_7e-05_128_3_22.pt"),
    ("ML1M_demographic","GMF_model"): Path(checkpoints_path, "GMF_model_ML1M_demographic_0.00061_64_21_10.pt"),
    ("ML1M_demographic","NCF"):Path(checkpoints_path, "NCF_ML1M_demographic_0.00023_32_3_2.pt"),

    
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,
    ("ML1M","LightGCN"): 8, #changed
    ("ML1M","MLP_model"): 8,
    ("ML1M","GMF_model"): 8,
    ("ML1M","NCF"): 8,

    ("ML1M_demographic","VAE"): None,
    ("ML1M_demographic","MLP"): 32,
    ("ML1M_demographic","MLP_model"): 8,
    ("ML1M_demographic","GMF_model"): 8,
    ("ML1M_demographic","NCF"): 8,
}

LXR_checkpoint_dict = {
    ("ML1M","VAE"): ('LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt',128),
    ("ML1M","LightGCN"): ('LXR_ML1M_LightGCN_0_39_64_11.59908096547193_0.1414854294885049.pt',64),#changed
    ("ML1M","MLP"): ('LXR_ML1M_MLP_12_39_64_11.59908096547193_0.1414854294885049.pt',64),
    ("ML1M","MLP_model"): 8,
    ("ML1M","GMF_model"): 8,
    ("ML1M","NCF"): ('LXR_ML1M_NCF_54_39_64_10.842340974213002_0.3962467608085718.pt', 64),

    ("ML1M_demographic","VAE"): ('LXR_ML1M_demographic_VAE_comb_0_28_128_4.336170186907191_1.7621772323665827.pt',128),
    ("ML1M_demographic","MLP"): 32,
    ("ML1M_demographic","MLP_model"): 8,
    ("ML1M_demographic","GMF_model"): 8,
    ("ML1M_demographic","NCF"): 8,
}

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 
demographic = demographic_dict[data_name]
if demographic:
    num_features = features_dict[data_name]
else:
    num_features = num_items_dict[data_name]
hidden_dim = hidden_dim_dict[(data_name,recommender_name)]

recommender_path = recommender_path_dict[(data_name,recommender_name)]
lxr_path = LXR_checkpoint_dict[(data_name,recommender_name)][0]
lxr_dim = LXR_checkpoint_dict[(data_name,recommender_name)][1]

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)
test_array = static_test_data.iloc[:,:-2].to_numpy()

with open(Path(files_path, f'jaccard_based_sim_{data_name}.pkl'), 'rb') as f:
    jaccard_dict = pickle.load(f) 

with open(Path(files_path, f'cosine_based_sim_{data_name}.pkl'), 'rb') as f:
    cosine_dict = pickle.load(f) 

with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f) 

with open(Path(files_path, f'tf_idf_dict_{data_name}.pkl'), 'rb') as f:
    tf_idf_dict = pickle.load(f) 

for i in range(num_features):
    for j in range(i, num_features):
        jaccard_dict[(j,i)]= jaccard_dict[(i,j)]
        cosine_dict[(j,i)]= cosine_dict[(i,j)]

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

kw_dict = {'device':device,
          'num_items': num_items,
          'demographic':demographic,
          'num_features':num_features,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name}

# # Recommenders Architecture

from ipynb.fs.defs.recommenders_architecture import *
importlib.reload(ipynb.fs.defs.recommenders_architecture)
from ipynb.fs.defs.recommenders_architecture import *


VAE_config= {
"enc_dims": [512,128],
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

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
import optuna
import logging
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, hidden_size, **kw):
        super(LightGCN, self).__init__()
        self.num_users = kw['num_users']
        self.num_items = kw['num_items']
        self.device = kw['device']
        self.embedding_dim = hidden_size
        self.n_layers = kw.get('n_layers', 3)
        self.alpha = kw.get('alpha', 0.1)  # Degree of propagation
        
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim).to(self.device)
        self.sigmoid = nn.Sigmoid()
        
        self.graph = self.build_graph(kw['user_item_matrix'])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def build_graph(self, user_item_matrix):
        user_item_matrix = sp.coo_matrix(user_item_matrix)
        rows, cols = user_item_matrix.row, user_item_matrix.col
        data = user_item_matrix.data
        
        adj = sp.coo_matrix((data, (rows, cols)), shape=(self.num_users + self.num_items, self.num_users + self.num_items))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        row_sum = np.array(adj.sum(1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv).tocoo()
        
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
    
    def forward(self, user_tensor, item_tensor):
        all_embeddings = self.get_embeddings()
        
        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]
        
        user_vec = user_embeddings[user_tensor]
        item_vec = item_embeddings[item_tensor]
        
        output = torch.matmul(user_vec, item_vec.T).to(self.device)
        return self.sigmoid(output).to(self.device)
    
    def get_embeddings(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        embeddings_list = [all_embeddings]
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.graph, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        lightgcn_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_embeddings = torch.mean(lightgcn_embeddings, dim=1)
        
        return lightgcn_embeddings

# create user-item matrix
# Load the ratings data
ratings = pd.read_csv('/home/amir/Documents/code/xrs/LXR/processed_data/ML1M/data files/ratings.dat', 
                      sep='::', header=None, engine='python', 
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# movies = pd.read_csv(
#     '/home/amir/Documents/code/xrs/LXR/processed_data/ML1M/data files/movies.dat',
#     engine='python',
#     delimiter='::',
#     names=['movieid', 'title', 'genre'],
#     index_col='movieid',
#     encoding='latin1' #fuckin hell
# )

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

print(user_item_matrix.shape)

def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)
    elif recommender_name=='LightGCN':
        # recommender = LightGCN(hidden_dim, **kw_dict)
        recommender = LightGCN(
            hidden_size=64,  # or any other hidden size you prefer
            num_users=num_users_dict["ML1M"],
            num_items=num_items_dict["ML1M"],
            device=kw_dict['device'],
            user_item_matrix=user_item_matrix,
            n_layers=3,
            alpha=0.1
                )
    elif recommender_name=='MLP_model':
        recommender = MLP_model(hidden_size=hidden_dim, num_layers=3, **kw_dict)
    elif recommender_name=='GMF_model':
        recommender = GMF_model(hidden_size=hidden_dim, **kw_dict)
    elif recommender_name=='NCF':
        MLP_temp = MLP_model(hidden_size=hidden_dim, num_layers=3, **kw_dict)
        GMF_temp = GMF_model(hidden_size=hidden_dim, **kw_dict)
        recommender = NCF(factor_num=hidden_dim, num_layers=3, dropout=0.5, model= 'NeuMF-pre', GMF_model= GMF_temp, MLP_model=MLP_temp, **kw_dict)
    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path), map_location=torch.device('cpu')) #changed
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    return recommender
    
recommender = load_recommender()

# # LXR definition and loading

class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size):
        super(Explainer, self).__init__()
        
        self.users_fc = nn.Linear(in_features = user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features = item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features = hidden_size*2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features=user_size).to(device),
            nn.Sigmoid()
        ).to(device)
        
        
    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(device)

        return expl_scores

class LXR_loss(nn.Module):
    def __init__(self, lambda_pos, lambda_neg, alpha):
        super(LXR_loss, self).__init__()
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha
        
        
    def forward(self, user_tensors, items_tensors, items_ids, pos_masks):
        neg_masks = torch.sub(torch.ones_like(pos_masks), pos_masks)
        x_masked_pos = user_tensors * pos_masks
        x_masked_neg = user_tensors * neg_masks
        x_masked_res_pos = recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict)
        x_masked_res_neg = recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict)
          
        pos_loss = -torch.mean(torch.log(x_masked_res_pos))
        neg_loss = torch.mean(torch.log(x_masked_res_neg))
        l1 = x_masked_pos[x_masked_pos>0].mean()
        combined_loss = self.lambda_pos*pos_loss + self.lambda_neg*neg_loss + self.alpha*l1
        
        return combined_loss, pos_loss, neg_loss, l1

def load_explainer(fine_tuning=False, lambda_pos=None, lambda_neg=None, alpha=None):
    explainer = Explainer(num_features, num_items, lxr_dim)
    lxr_checkpoint = torch.load(Path(checkpoints_path, lxr_path), map_location=torch.device('cpu')) #changed
    explainer.load_state_dict(lxr_checkpoint)
    if not fine_tuning:
        explainer.eval()
        for param in explainer.parameters():
            param.requires_grad= False
        return explainer
    else:
        lxr_loss = LXR_loss(lambda_pos, lambda_neg, alpha)
        return explainer, lxr_loss
    
explainer = load_explainer(False)

# # Help functions

from ipynb.fs.defs.help_functions import *
importlib.reload(ipynb.fs.defs.help_functions)
from ipynb.fs.defs.help_functions import *

# # Baselines functions

from ipynb.fs.defs.lime import distance_to_proximity, LimeBase, get_lime_args, gaussian_kernel#, get_lire_args #changed
importlib.reload(ipynb.fs.defs.lime)
from ipynb.fs.defs.lime import distance_to_proximity, LimeBase, get_lime_args, gaussian_kernel#, get_lire_args #changed

lime = LimeBase(distance_to_proximity)

#User based similarities using Jaccard
def find_jaccard_mask(x, item_id, user_based_Jaccard_sim):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in user_based_Jaccard_sim:
                item_jaccard_dict[i]=user_based_Jaccard_sim[(i,item_id)] # add Jaccard similarity between items
            else:
                item_jaccard_dict[i] = 0            

    return item_jaccard_dict

#Cosine based similarities between users and items
def find_cosine_mask(x, item_id, item_cosine):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in item_cosine:
                item_cosine_dict[i]=item_cosine[(i,item_id)]
            else:
                item_cosine_dict[i]=0

    return item_cosine_dict

#popularity mask
def find_pop_mask(x, item_id):
    user_hist = torch.Tensor(x).to(device) # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_pop_dict = {}
    
    for i,j in enumerate(user_hist>0):
        if j:
            item_pop_dict[i]=pop_array[i] # add the pop of the item to the dictionary
            
    return item_pop_dict

def find_lime_mask(x, item_id, min_pert, max_pert, num_of_perturbations, kernel_func, feature_selection, recommender, num_samples=10, method = 'POS', **kw_dict):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    lime.kernel_fn = kernel_func
    neighborhood_data, neighborhood_labels, distances, item_id = get_lime_args(user_hist, item_id, recommender, all_items_tensor, min_pert = min_pert, max_pert = max_pert, num_of_perturbations = num_of_perturbations, seed = item_id, **kw_dict)
    if method=='POS':
        most_pop_items  = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_samples, feature_selection, pos_neg='POS')
    if method=='NEG':
        most_pop_items  = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_samples, feature_selection ,pos_neg='NEG')
        
    return most_pop_items 

def find_lire_mask(x, item_id, num_of_perturbations, kernel_func, feature_selection, recommender, proba=0.1, method = 'POS', **kw_dict):
    
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    lime.kernel_fn = kernel_func
    neighborhood_data, neighborhood_labels, distances, item_id = get_lire_args(user_hist, item_id, recommender, all_items_tensor, train_array, num_of_perturbations = num_of_perturbations, seed = item_id, proba=0.1, **kw_dict)
    if method=='POS':
        most_pop_items  = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_of_perturbations, feature_selection, pos_neg='POS')
    if method=='NEG':
        most_pop_items  = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_of_perturbations, feature_selection ,pos_neg='NEG')
        
    return most_pop_items

def find_tf_idf_mask(x, item_id, tf_idf_sim, user_id):

    x = x.cpu().detach().numpy()
    x[item_id] = 0
    user_tf_idf_scores = tf_idf_sim[user_id].copy()
  
    return user_tf_idf_scores

def find_fia_mask(user_tensor, item_tensor, item_id, recommender):
    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(device)
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in range(num_items):
        if(user_hist[i] == 1):
            user_hist[i] = 0
            user_tensor = torch.FloatTensor(user_hist).to(device)
            y_pred_without_item = recommender_run(user_tensor, recommender, item_tensor, item_id, 'single', **kw_dict).to(device)
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = infl_score
            user_hist[i] = 1

    return items_fia

def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k):
   
    items_accent = defaultdict(float)
    factor = top_k - 1
    user_accent_hist = user_tensor.cpu().detach().numpy().astype(int)

    #Get topk items
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    
    if top_k == 1:
        # When k=1, return the index of the first maximum value
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]
   

    for iteration, item_k_id in enumerate(top_k_indices):

        # Set topk items to 0 in the user's history
        user_accent_hist[item_k_id] = 0
        user_tensor = torch.FloatTensor(user_accent_hist).to(device)
       
        item_vector = items_array[item_k_id]
        item_tensor = torch.FloatTensor(item_vector).to(device)
              
        # Check influence of the items in the history on this specific item in topk
        fia_dict = find_fia_mask(user_tensor, item_tensor, item_k_id, recommender_model)
         
        # Sum up all differences between influence on top1 and other topk values
        if not iteration:
            for key in fia_dict.keys():
                items_accent[key] *= factor
        else:
            for key in fia_dict.keys():
                items_accent[key] -= fia_dict[key]
       
    for key in items_accent.keys():
        items_accent[key] *= -1    

    return items_accent

def find_lxr_mask(x, item_tensor):
    
    user_hist = x
    expl_scores = explainer(user_hist, item_tensor)
    x_masked = user_hist*expl_scores
    item_sim_dict = {}
    for i,j in enumerate(x_masked>0):
        if j:
            item_sim_dict[i]=x_masked[i] 
        
    return item_sim_dict

def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender_model, expl_dict, **kw_dict):
    POS_masked = user_tensor
    NEG_masked = user_tensor
    POS_masked[item_id]=0
    NEG_masked[item_id]=0
    user_hist_size = np.sum(user_vector)
    
    
    bins=[0]+[len(x) for x in np.array_split(np.arange(user_hist_size), num_of_bins, axis=0)]
    
    POS_at_1 = [0]*(len(bins))
    POS_at_5 = [0]*(len(bins))
    POS_at_10=[0]*(len(bins))
    POS_at_20=[0]*(len(bins))
    POS_at_50=[0]*(len(bins))
    POS_at_100=[0]*(len(bins))
    
    NEG_at_1 = [0]*(len(bins))
    NEG_at_5 = [0]*(len(bins))
    NEG_at_10 = [0]*(len(bins))
    NEG_at_20 = [0]*(len(bins))
    NEG_at_50 = [0]*(len(bins))
    NEG_at_100 = [0]*(len(bins))
    
    DEL = [0]*(len(bins))
    INS = [0]*(len(bins))
    
    rankA_at_1 = [0]*(len(bins))
    rankA_at_5 = [0]*(len(bins))
    rankA_at_10 = [0]*(len(bins))
    rankA_at_20 = [0]*(len(bins))
    rankA_at_50 = [0]*(len(bins))
    rankA_at_100 = [0]*(len(bins))
    
    rankB = [0]*(len(bins))
    NDCG = [0]*(len(bins))

    
    POS_sim_items = expl_dict
    NEG_sim_items  = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1],reverse=False))
    
    total_items=0
    for i in range(len(bins)):
        total_items += bins[i]
            
        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        
        for j in POS_sim_items[:total_items]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor - POS_masked # remove the masked items from the user history

        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in NEG_sim_items[:total_items]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor - NEG_masked # remove the masked items from the user history 
        
        POS_ranked_list = get_top_k(POS_masked, user_tensor, recommender_model, **kw_dict)
        
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id)+1
        else:
            POS_index = num_items
        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender_model, **kw_dict)+1

        # for pos:
        POS_at_1[i] = 1 if POS_index <=1 else 0
        POS_at_5[i] = 1 if POS_index <=5 else 0
        POS_at_10[i] = 1 if POS_index <=10 else 0
        POS_at_20[i] = 1 if POS_index <=20 else 0
        POS_at_50[i] = 1 if POS_index <=50 else 0
        POS_at_100[i] = 1 if POS_index <=100 else 0

        # for neg:
        NEG_at_1[i] = 1 if NEG_index <=1 else 0
        NEG_at_5[i] = 1 if NEG_index <=5 else 0
        NEG_at_10[i] = 1 if NEG_index <=10 else 0
        NEG_at_20[i] = 1 if NEG_index <=20 else 0
        NEG_at_50[i] = 1 if NEG_index <=50 else 0
        NEG_at_100[i] = 1 if NEG_index <=100 else 0

        # for del:
        DEL[i] = float(recommender_run(POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        # for ins:
        INS[i] = float(recommender_run(user_tensor-POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        # for rankA:
        rankA_at_1[i] = max(0, (1+1-POS_index)/1)
        rankA_at_5[i] = max(0, (5+1-POS_index)/5)
        rankA_at_10[i] = max(0, (10+1-POS_index)/10)
        rankA_at_20[i] = max(0, (20+1-POS_index)/20)
        rankA_at_50[i] = max(0, (50+1-POS_index)/50)
        rankA_at_100[i] = max(0, (100+1-POS_index)/100)

        # for rankB:
        rankB[i] = 1/POS_index

        #for NDCG:
        NDCG[i]= get_ndcg(list(POS_ranked_list.keys()),item_id, **kw_dict)
        
    res = [DEL, INS, rankB, NDCG, POS_at_1, POS_at_5, POS_at_10, POS_at_20, POS_at_50, POS_at_100,  NEG_at_1, NEG_at_5, NEG_at_10, NEG_at_20, NEG_at_50, NEG_at_100,  rankA_at_1, rankA_at_5, rankA_at_10, rankA_at_20, rankA_at_50, rankA_at_100]
    for i in range(len(res)):
        res[i] = np.mean(res[i])
        
    return res

# # usecases functions

def print_items(df):
    x = PrettyTable()
    x.field_names = ["item id", "movie name", "year", "popularity"]

    for row in df.iterrows():
        x.add_row([row[1][1],row[1][0],row[1][-1], pop_dict[row[1][1]]])
    print(x)


def create_df_by_genre(genre_name):
    return movies[movies[genre_name]==1]


def return_expl_scores(user_tens):
    recommended_item_id = get_user_recommended_item(user_tens, recommender, **kw_dict)
    recommended_item_tensor = torch.zeros(num_items).to(device)
    recommended_item_tensor[recommended_item_id] = 1 
    expl_scores = find_lxr_mask(user_tens, recommended_item_tensor)
    sorted_list = sorted(expl_scores.items(), key=lambda x:x[1], reverse=True)
    print(f'Explanation scores for the movie {movies.iloc[recommended_item_id.item()]}:\n')
    L = []
    for i in sorted_list:
        item_id = i[0]
        L.append(item_id)
        item_score = i[1].item()
        print(round(item_score, 5), movies.iloc[item_id][0])
    print(movies.loc[L])

def print_user(user_vec):
    ids = np.where(user_vec==1)
    small_df = movies.loc[ids]
    print(small_df)

# This function does ITF and return the best metric value
def lxr_itf_usecases(user_vector, item_id, epochs, L_pos, L_neg, alpha, learning_rate):
    
    itf_explainer, loss_func = load_explainer(True, L_pos, L_neg, alpha)
    optimizer = torch.optim.Adam(itf_explainer.parameters(), lr=learning_rate)
    user_hist = torch.Tensor(user_vector).to(device)
    user_size = np.sum(user_vector>0)
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    rows=[]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        expl_scores = itf_explainer(user_hist, item_tensor)
        
        x_masked = user_hist*expl_scores
        comb_loss, pos_loss, neg_loss, l1 = loss_func(user_hist, item_tensor, item_id, expl_scores)
        comb_loss.backward()
        optimizer.step()        
        
        # get the metrics values for the current step and add it to the results array
        item_sim_dict = {}
        for i,j in enumerate(user_hist>0):
            if j:
                item_sim_dict[i]=x_masked[i].item()
        rows.append(item_sim_dict)

    item_sim_dict = {}
    expl_scores = itf_explainer(user_hist, item_tensor)
    x_masked = user_hist*expl_scores
    item_sim_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            item_sim_dict[i]=x_masked[i].item()

    rows.append(item_sim_dict)
    
    expl_df = pd.DataFrame(rows,columns=list(np.argwhere(user_vector>0)).append(item_id))
    
    return expl_df

user_vector = np.zeros(num_items)
childrens_movies = [0, 1, 335, 543, 550, 868, 1558, 2268, 506,541, 333, 442, 766]
war_horror = [1,759, 1014, 103, 924, 1019, 815, 1119, 31, 442, 561]
childern_war = [1, 442, 236, 601, 766, 1010, 996, 927, 1786 ]
boysgirls = [0, 333, 335, 543, 549, 861, 871, 1731, 1727, 541, 550]
romance_drama = [37, 16, 24, 266, 327, 328, 551, 34, 103, 145]
user_vector[childrens_movies]=1

user_tensor = torch.Tensor(user_vector).to(device)
recommended_item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
recommended_item_tensor = torch.zeros(num_items).to(device)
recommended_item_tensor[recommended_item_id] = 1 
print(movies.iloc[recommended_item_id][0])

recommended_item_id

movies.iloc[recommended_item_id]

epochs = 10
start_time = time.time()
item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
res_df = lxr_itf_usecases(user_vector,item_id, epochs, 3.185652725834087, 1.420642300151429,1, 0.01)

prev_time = time.time()
print("User {}, total time: {:.2f}".format(i,prev_time - start_time))


list(movies.columns[(movies.iloc[i]==1)])

res_df.rename(columns={i:f'{movies.iloc[i][0]}' for i in res_df.columns}, inplace=True)
# res_df.rename(columns={i:f'{movies.iloc[i][0]}, {i}, {sorted(list(movies.columns[(movies.iloc[i]==1)]))}' for i in res_df.columns}, inplace=True)

import seaborn as sns
sns.heatmap(res_df, cmap='Reds', square=True, vmax=1.05)
plt.title(f"Explanation Scores for {movies.iloc[recommended_item_id][0]}During ITF", fontsize =12)
plt.ylabel('Fine Tuning Step')
plt.savefig(Path(export_dir,f'LXR_ITF_explanation_{movies.iloc[recommended_item_id][0]}.png'), dpi=500, bbox_inches='tight', pad_inches=0)


# Children = create_df_by_genre('Children\'s')

# print_items(Children)

# print_items(Romance)

movies

user_vector = np.zeros(num_items)
childrens_movies = [0, 1, 335, 543, 550, 868, 1558, 2268, 1968]
user_vector[childrens_movies]=1

user_tens = torch.Tensor(user_vector).to(device)
recommended_item_id = int(get_user_recommended_item(user_tens, recommender, **kw_dict))
recommended_item_tensor = torch.zeros(num_items).to(device)
recommended_item_tensor[recommended_item_id] = 1 
print(movies.iloc[recommended_item_id][0])

# This function does ITF and plots all items explanations scores per step
# It also plots the sum of explanation scores per step


def find_lxr_itf_mask_plot_expl_scores(user_vector, item_id, epochs, L_pos, L_neg, alpha):
    
    itf_explainer, loss_func = load_explainer(True, L_pos, L_neg, alpha)
    optimizer = torch.optim.Adam(itf_explainer.parameters(), lr=0.001)
    user_hist = torch.Tensor(user_vector).to(device)
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    expl_strength = []
    
    epochs_range = np.arange(epochs)
    items_names = [movies.iloc[item_id][0] for item_id in np.where(user_vector==1)[0]]
    expl_array = np.zeros((epochs, sum(user_vector)))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        expl_scores = itf_explainer(user_hist, item_tensor)
        
        x_masked = user_hist*expl_scores
        comb_loss, pos_loss, neg_loss, l1 = loss_func(user_hist, item_tensor, item_id, expl_scores)
        comb_loss.backward()
        optimizer.step()
        expl_strength.append(sum(x_masked.cpu()).item())
        
        k=0
        for i,j in enumerate(user_hist>0):
            if j:
                expl_array[epoch,k] = x_masked[i]
                k+=1
        

    # set width of bar 
    barWidth = 1/(epochs+5)
    fig = plt.subplots(figsize =(12, 8)) 

    # Set position of bar on X axis 
    br1 = np.arange( sum(user_vector).item()) 

    for epoch in range(epochs):
        br = [x + barWidth*epoch for x in br1] 
        scores = expl_array[epoch,:]
        
        # Make the plot
        plt.bar(br, scores, width = barWidth, 
                edgecolor ='grey') 

    # Adding Xticks 
    plt.xlabel('item', fontsize = 15) 
    plt.ylabel('scores', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(sum(user_vector))], 
            items_names, rotation=90)
    plt.show()
        
    
    item_sim_dict = {}
    expl_scores = itf_explainer(user_hist, item_tensor)
    with torch.no_grad():
        x_masked = user_hist*expl_scores
        item_sim_dict = {}
        for i,j in enumerate(user_hist>0):
            if j:
                item_sim_dict[i]=x_masked[i] 
        
    plt.plot(expl_strength)
    
    return 

user_vec = test_array[885]
item_id = get_user_recommended_item(torch.Tensor(user_vec).to(device), recommender, **kw_dict).item()
# print(f'Explaining {movies.iloc[item_id][0]}')
def find_lxr_itf_mask_plot_metrics(user_vector, item_id, epochs, L_pos, L_neg, alpha):
    
    itf_explainer, loss_func = load_explainer(True, L_pos, L_neg, alpha)
    optimizer = torch.optim.Adam(itf_explainer.parameters(), lr=0.001)
    user_hist = torch.Tensor(user_vector).to(device)
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    
    epochs_range = np.arange(epochs)
    expl_array = np.zeros((epochs+1, 22))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        expl_scores = itf_explainer(user_hist, item_tensor)
        
        x_masked = user_hist*expl_scores
        comb_loss, pos_loss, neg_loss, l1 = loss_func(user_hist, item_tensor, item_id, expl_scores)
        comb_loss.backward()
        optimizer.step()        
        
        item_sim_dict = {}
        for i,j in enumerate(user_hist>0):
            if j:
                item_sim_dict[i]=x_masked[i] 
        expl_dict  = list(sorted(item_sim_dict.items(), key=lambda item: item[1],reverse=True))
        expl_array[epoch,:] = single_user_metrics(user_vector, user_hist, item_id, item_tensor, 10, recommender, expl_dict, **kw_dict)


    item_sim_dict = {}
    expl_scores = itf_explainer(user_hist, item_tensor)
    with torch.no_grad():
        x_masked = user_hist*expl_scores
        item_sim_dict = {}
        for i,j in enumerate(user_hist>0):
            if j:
                item_sim_dict[i]=x_masked[i] 
        expl_dict  = list(sorted(item_sim_dict.items(), key=lambda item: item[1],reverse=True))
        expl_array[epoch+1,:] = single_user_metrics(user_vector, user_hist, item_id, item_tensor, 10, recommender, expl_dict, **kw_dict)
                
                
    # set width of bar 
    barWidth = 1/(epochs+5)
    fig = plt.subplots(figsize =(12, 8)) 

    # Set position of bar on X axis 
    br1 = np.arange(22) 

    for epoch in range(epochs+1):
        br = [x + barWidth*epoch for x in br1] 
        scores = expl_array[epoch,:]
        
        # Make the plot
        plt.bar(br, scores, width = barWidth, 
                edgecolor ='grey') 

    # Adding Xticks 
    plt.xlabel('metrics', fontsize = 15) 
    plt.ylabel('scores', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(22)], 
            ['users_DEL', 'users_INS', 'reciprocal', 'NDCG', 'POS_at_1', 'POS_at_5', 'POS_at_10', 'POS_at_20', 'POS_at_50', 'POS_at_100', 'NEG_at_1', 'NEG_at_5', 'NEG_at_10', 'NEG_at_20', 'NEG_at_50', 'NEG_at_100', 'rank_at_1', 'rank_at_5', 'rank_at_10', 'rank_at_20', 'rank_at_50', 'rank_at_100'], rotation=90)

    plt.show() 

    expl_result = expl_array.T
    result = [0]*22
    for i in range(22):
        if i in [1,5,8,11,14,17,20]:
            result[i] = max(expl_result[i,:])
        else:
            result[i] = min(expl_result[i,:])
    return result 

find_lxr_itf_mask_plot_metrics(user_vec,item_id, 10,10, 0.4, 1)

# This function does ITF and plots all metrics per epoch 
# It also return the best value for every metric


user_vec = test_array[885]
item_id = get_user_recommended_item(torch.Tensor(user_vec).to(device), recommender, **kw_dict).item()
print(f'Explaining {movies.iloc[item_id][0]}')
find_lxr_itf_mask_plot_expl_scores(user_vec,item_id, 20,10.84234, 0.4, 1)





# # Create and explain usecases



# This function does ITF and return the best metric value
def find_lxr_itf_mask(user_vector, item_id, epochs, L_pos, L_neg, alpha):
    
    itf_explainer, loss_func = load_explainer(True, L_pos, L_neg, alpha)
    optimizer = torch.optim.Adam(itf_explainer.parameters(), lr=0.001)
    user_hist = torch.Tensor(user_vector).to(device)
    item_tensor = torch.Tensor(items_array[item_id]).to(device)
    expl_array = np.zeros((epochs+1, 22))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        expl_scores = itf_explainer(user_hist, item_tensor)
        
        x_masked = user_hist*expl_scores
        comb_loss, pos_loss, neg_loss, l1 = loss_func(user_hist, item_tensor, item_id, expl_scores)
        comb_loss.backward()
        optimizer.step()        
        
        item_sim_dict = {}
        for i,j in enumerate(user_hist>0):
            if j:
                item_sim_dict[i]=x_masked[i] 
        expl_dict  = list(sorted(item_sim_dict.items(), key=lambda item: item[1],reverse=True))
        expl_array[epoch,:] = single_user_metrics(user_vector, user_hist, item_id, item_tensor, 10, recommender, expl_dict, **kw_dict)

    item_sim_dict = {}
    expl_scores = itf_explainer(user_hist, item_tensor)
    with torch.no_grad():
        x_masked = user_hist*expl_scores
        item_sim_dict = {}
        for i,j in enumerate(user_hist>0):
            if j:
                item_sim_dict[i]=x_masked[i] 
        expl_dict  = list(sorted(item_sim_dict.items(), key=lambda item: item[1],reverse=True))
        expl_array[epoch+1,:] = single_user_metrics(user_vector, user_hist, item_id, item_tensor, 10, recommender, expl_dict, **kw_dict)
                
    expl_result = expl_array.T
    result = [0]*22
    for i in range(22):
        if i in [1,5,8,11,14,17,20]:
            result[i] = max(expl_result[i,:])
        else:
            result[i] = min(expl_result[i,:])
    return np.array(result) 

res = np.zeros(22)
time1 = time.time()
for i in range(2):
    user_vec = test_array[i]
    item_id = get_user_recommended_item(torch.Tensor(user_vec).to(device), recommender, **kw_dict).item()
    res+=find_lxr_itf_mask(user_vec,item_id, 10,7, 0.4, 1)
    print(res)
res/(i+1)
print(time.time()-time1)






