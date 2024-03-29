'''
@desc: take the pseudo-code, the control-flow graph (CFG), and function-level features as input
       import the pretrained semantic encoder and frozen it, only train the graph encoder
@usage:  python3 5_train_gnn.py
    @input_file: 
        ../cryptobench/GCN/train.csv
        ../cryptobench/GCN/train_all_feature.json
        ../cryptobench/GCN/test.csv
        ../cryptobench/GCN/test_all_feature.json
'''

import os
import torch
import wandb
import json
import pickle
import random
import logging
import pandas as pd 
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch.optim import Adam
from typing import List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import tokenization_utils, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from GNN.graphembeddingnetwork import get_default_config, build_model, GraphEmbeddingNet
from sklearn.metrics import accuracy_score
from utils.utils import str_to_scipy_sparse

logging.basicConfig(level=logging.INFO,format="%(asctime)s| %(levelname)s| %(message)s")
tokenization_utils.logger.setLevel('ERROR')
random.seed(233)

# <parameters start>
model_name = "gnn-model"
model_save_path = f"../models/chkp_{model_name}"
resume_from_checkpoint = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_group_path, train_feature_path = "../cryptobench/GCN/train.csv","../cryptobench/GCN/train_all_feature.json"
test_group_path, test_feature_path = "../cryptobench/GCN/test.csv","../cryptobench/GCN/test_all_feature.json"
train_batch_size = 48
eval_batch_size = 48
gradient_accumulation_steps = 1
learning_rate = 1e-3
weight_decay = 1e-5
train_total_steps = 80000
eval_interval_steps = train_total_steps//10
wandb_mode = "online" # "online", "offline"
# <parameters end>

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    logging.info(f"\n[-] compute metrics, get predictions {len(predictions)}\n")
    return {'accuracy': accuracy_score(labels, predictions)}

class MultipleNegativesRankingLoss:
    def __init__(self):
        super()

    def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def __call__(self, anchor_embs: List[torch.Tensor], pos_embs: List[torch.Tensor], scale: float = 20.0):
        scores = self.cos_sim(anchor_embs, pos_embs) * scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return torch.nn.functional.cross_entropy(scores, labels)

class GraphModelForACFG(GraphEmbeddingNet):
    def reshape_and_split_tensor(self, tensor, n_splits):
        """Reshape and split a 2D tensor along the last dimension.

        Args:
        tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
            multiple of `n_splits`.
        n_splits: int, number of splits to split the tensor into.

        Returns:
        splits: a list of `n_splits` tensors.  The first split is [tensor[0],
            tensor[n_splits], tensor[n_splits * 2], ...], the second split is
            [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
        """
        feature_dim = tensor.shape[-1]
        tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
        tensor_split = []
        for i in range(n_splits):
            tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
        return tensor_split
    
    def forward(self,
                node_features,
                edge_features,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                pcode,
                fea_embed,
                labels: Optional[torch.LongTensor] = None,
                generate_emb: bool = False,
                ):
        node_features, edge_features = self._graphencoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]

        for layer in self._prop_layers:
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                edge_features)
            layer_outputs.append(node_states)

        self._layer_outputs = layer_outputs
        graph_embs = self._aggregator(node_states, graph_idx, n_graphs)
        
        with torch.no_grad():
            pcode_embed = self._codet5encoder(pcode) 

        graph_embs = torch.cat((graph_embs, pcode_embed),dim=1)
        graph_embs = torch.cat((graph_embs, fea_embed),dim=1)
        fusion_embs = self._fusionembedder(graph_embs)

        if generate_emb:
            return fusion_embs
    
        anchor_embs, query_embs = self.reshape_and_split_tensor(fusion_embs, 2)

        preds = None
        loss_fct = MultipleNegativesRankingLoss()
        multi_neg_ranking_loss = loss_fct(anchor_embs, query_embs)
        if labels is None: # train
            pass
        else: # valid or test
            scores = loss_fct.cos_sim(anchor_embs, query_embs)
            preds = scores.argmax(dim=-1)
        
        return SequenceClassifierOutput(
            loss=multi_neg_ranking_loss,
            logits=preds,
        )

class ACFGDataset(Dataset):
    def __init__(self, function_group_path, feature_path,) -> None:
        super().__init__()
        self.features = []
        self.func_pairs = []
        self._load_data(func_path=function_group_path, feat_path=feature_path)
        random.shuffle(self.func_pairs)

    def __len__(self) -> int:
        return len(self.func_pairs)
    
    def __getitem__(self, index) -> Any:
        anchor_func, pos_func = self.func_pairs[index]
        anchor_graph = nx.DiGraph(str_to_scipy_sparse(self.features[str(anchor_func)]['graph']))
        anchor_features = str_to_scipy_sparse(self.features[str(anchor_func)]['opc'])
        anchor_edges = np.array(anchor_graph.edges, dtype=np.int32)
        anchor_pcode = self.features[str(anchor_func)]['pcode']
        anchor_fea_embed = self.features[str(anchor_func)]['feature']

        pos_graph = nx.DiGraph(str_to_scipy_sparse(self.features[str(pos_func)]['graph']))
        pos_features = str_to_scipy_sparse(self.features[str(pos_func)]['opc'])
        pos_edges = np.array(pos_graph.edges, dtype=np.int32)
        pos_pcode = self.features[str(pos_func)]['pcode']
        pos_fea_embed = self.features[str(pos_func)]['feature']
        
        anchor = {
            'node_features':np.float32(anchor_features),
            'edge_features':np.zeros((len(anchor_edges),1),dtype=np.float32),
            'from_idx':anchor_edges[:,0], 
            'to_idx':anchor_edges[:,1], 
            'pcode':anchor_pcode,
            'fea_embed':np.float32(anchor_fea_embed)
        }
        pos = {
            'node_features':np.float32(pos_features),
            'edge_features':np.zeros((len(pos_edges),1),dtype=np.float32),
            'from_idx':pos_edges[:,0], 
            'to_idx':pos_edges[:,1],
            'pcode':pos_pcode,
            'fea_embed':np.float32(pos_fea_embed)
        }
        return anchor, pos
    
    def _load_data(self, func_path, feat_path):
        # Load CSV with the list of functions
        print("Reading {}".format(func_path))
        # Read the CSV and reset the index
        self._df_func = pd.read_csv(func_path)
        fid_list = self._df_func["fid"]
        func_name_list = self._df_func["func_name"]

        from collections import defaultdict
        # Get the list of indexes associated to each function name
        self._func_name_dict = defaultdict(set)
        for i,key in enumerate(func_name_list) :
            self._func_name_dict[key].add(fid_list[i])
        
        # Get the list of unique function name
        self._func_name_list = list(self._func_name_dict.keys())
        print("Found {} functions".format(len(self._func_name_list)))

        for fname in tqdm(self._func_name_list, desc="collect func pairs"):
            group = list(self._func_name_dict[fname])
            if len(group) < 2:
                continue
            for func_iloc in group:
                pos_func_iloc = random.choice(group)
                while func_iloc == pos_func_iloc:
                    pos_func_iloc = random.choice(group)
                self.func_pairs.append((func_iloc,pos_func_iloc))

        # Load the JSON with functions features
        print("Loading {}".format(feat_path))
        with open(feat_path, "r", encoding="utf-8") as f:
            self.features = json.load(f)

class ACFGDatasetForGenEmb(Dataset):
    def __init__(self, function_group_path, feature_path,) -> None:
        super().__init__()
        self.function_group_path = function_group_path
        self.feature_path = feature_path
        self.funcs = []
        self._load_data(func_path=function_group_path, feat_path=feature_path)

    def __len__(self) -> int:
        return len(self.funcs)
    
    def __getitem__(self, index) -> Any:
        func_dict = self.funcs[index]
        fid = func_dict['fid']
        anchor_graph = nx.DiGraph(str_to_scipy_sparse(self.features[str(fid)]['graph']))
        anchor_features = str_to_scipy_sparse(self.features[str(fid)]['opc'])
        anchor_edges = np.array(anchor_graph.edges, dtype=np.int32)
        anchor_pcode = self.features[str(fid)]['pcode']
        anchor_fea_embed = self.features[str(fid)]['feature']
        
        anchor = {
            'node_features':np.float32(anchor_features),
            'edge_features':np.zeros((len(anchor_edges),1),dtype=np.float32),
            'from_idx':anchor_edges[:,0], 
            'to_idx':anchor_edges[:,1],
            'fid':fid,
            'pcode':anchor_pcode,
            'fea_embed':np.float32(anchor_fea_embed)
        }
        return (anchor,)
    
    def _load_data(self, func_path, feat_path):
        # Load CSV with the list of functions
        print("Reading {}".format(func_path))
        # Read the CSV and reset the index
        self._df_func = pd.read_csv(func_path)
        fid_list = self._df_func["fid"]
        func_name_list = self._df_func["func_name"]
        from collections import defaultdict
        # Get the list of indexes associated to each function name
        self._func_name_dict = defaultdict(set)
        for i,key in enumerate(func_name_list) :
            self._func_name_dict[key].add(fid_list[i])
        
        # Get the list of unique function name
        self._func_name_list = list(self._func_name_dict.keys())
        print("Found {} function groups".format(len(self._func_name_list)))

        for fname in tqdm(self._func_name_list, desc="get func list"):
            group = list(self._func_name_dict[fname])
            for func_iloc in group:
                self.funcs.append({
                    "fid":func_iloc
                    })
                    
        # Load the JSON with functions features
        print("Loading {}".format(feat_path))
        with open(feat_path, "r", encoding="utf-8") as f:
            self.features = json.load(f)

def collate_fn(batch):
    graphs = []
    for pair in batch:
        for intergraph in pair:
            graphs.append(intergraph)

    node_features = []
    edge_features = []
    from_idx = []
    to_idx = []
    graph_idx = []
    pcode = []
    fea_embed = []
    n_total_nodes = 0  
    n_total_edges = 0
    for idx, graph in enumerate(graphs):
        node_features.append(graph['node_features'])
        edge_features.append(graph['edge_features'])
        n_nodes = len(graph['node_features'])
        n_edges = len(graph['edge_features'])
        # shift the node indices for the edges
        from_idx.append(graph['from_idx'] + n_total_nodes)
        to_idx.append(graph['to_idx'] + n_total_nodes)
        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * idx)
        pcode.append(graph['pcode'])
        fea_embed.append(graph['fea_embed'])

        n_total_nodes += n_nodes
        n_total_edges += n_edges
    
    node_features = torch.from_numpy(np.concatenate(node_features, axis=0))
    edge_features = torch.from_numpy(np.concatenate(edge_features, axis=0))
    from_idx = torch.from_numpy(np.concatenate(from_idx, axis=0)).long()
    to_idx = torch.from_numpy(np.concatenate(to_idx, axis=0)).long()
    graph_idx = torch.from_numpy(np.concatenate(graph_idx, axis=0)).long()
    labels = torch.tensor(range(len(batch))).long() # MultipleNegativesRankingLoss labels
    pcode = pcode
    fea_embed = torch.tensor(np.array(fea_embed))

    return {
            'node_features':node_features,
            'edge_features':edge_features,
            'from_idx':from_idx,
            'to_idx':to_idx,
            'graph_idx':graph_idx,
            'n_graphs':len(graphs),
            'labels':labels,
            'pcode':pcode,
            'fea_embed':fea_embed
        }

def collate_fn_forGenEmb(batch):
    graphs = []
    for pair in batch:
        for intergraph in pair:
            graphs.append(intergraph)

    node_features = []
    edge_features = []
    from_idx = []
    to_idx = []
    graph_idx = []
    fids = []
    pcode = []
    fea_embed = []
    n_total_nodes = 0
    n_total_edges = 0
    for idx, graph in enumerate(graphs):
        node_features.append(graph['node_features'])
        edge_features.append(graph['edge_features'])
        n_nodes = len(graph['node_features'])
        n_edges = len(graph['edge_features'])
        # shift the node indices for the edges
        from_idx.append(graph['from_idx'] + n_total_nodes)
        to_idx.append(graph['to_idx'] + n_total_nodes)
        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * idx)
        fids.append(graph['fid'])
        pcode.append(graph['pcode'])
        fea_embed.append(graph['fea_embed'])
        n_total_nodes += n_nodes
        n_total_edges += n_edges
    
    node_features = torch.from_numpy(np.concatenate(node_features, axis=0))
    edge_features = torch.from_numpy(np.concatenate(edge_features, axis=0))
    from_idx = torch.from_numpy(np.concatenate(from_idx, axis=0)).long()
    to_idx = torch.from_numpy(np.concatenate(to_idx, axis=0)).long()
    graph_idx = torch.from_numpy(np.concatenate(graph_idx, axis=0)).long()
    labels = torch.tensor(range(len(batch))).long() # MultipleNegativesRankingLoss labels
    pcode = pcode
    fea_embed = torch.tensor(np.array(fea_embed))
    
    return {
            'node_features':node_features.to(device),
            'edge_features':edge_features.to(device),
            'from_idx':from_idx.to(device),
            'to_idx':to_idx.to(device),
            'graph_idx':graph_idx.to(device),
            'n_graphs':len(graphs),
            'labels':labels,
            'fids':fids,
            'pcode':pcode,
            'fea_embed':fea_embed.to(device)
        }

def train(model_path:str = None):
    wandb.init(project="Train_GNN", name=model_save_path, resume=resume_from_checkpoint, mode=wandb_mode)
    model_path = model_path if model_path else model_name
    logging.info(f"[-] training model {model_path}")
    config = get_default_config()
    model = build_model(config,GraphModelForACFG,node_feature_dim=200,edge_feature_dim=1)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # read data
    logging.info("[-] reading train and test data...")
    train_dataset = ACFGDataset(train_group_path, train_feature_path)
    test_dataset = ACFGDataset(test_group_path, test_feature_path)
    logging.info(f"[-] train dataset len {len(train_dataset)}, test dataset len {len(test_dataset)}")

    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=8,
        logging_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        max_steps=train_total_steps,
        save_strategy="steps",
        save_steps=eval_interval_steps,
        evaluation_strategy="steps",
        eval_steps=eval_interval_steps,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=10, early_stopping_threshold=0.0000001
            )],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def generate_embedding(model_state_path, group_path, feature_path):
    logging.info(f"[-] loading model from {model_state_path}")
    model_state = torch.load(model_state_path)
    config = get_default_config()
    model = build_model(config, GraphModelForACFG, 
                        node_feature_dim=200, edge_feature_dim=1)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    func_embs = dict()
    dataset = ACFGDatasetForGenEmb(group_path, feature_path)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn_forGenEmb)
    for batch in tqdm(dataloader, desc="generate embedding"):
        fids = batch.pop('fids')
        embs = model(**batch, generate_emb=True)
        for idx, fid in enumerate(fids):
            func_embs[fid] = embs[idx].detach().cpu().tolist()
    
    save_to = os.path.splitext(group_path)[0] + ".embs.pkl"
    with open(save_to,"wb") as f:
        pickle.dump(func_embs,f)

if __name__ == "__main__":
    
    # 1. train on test dataset
    # train()

    # 2. eval on test dataset
    generate_embedding(model_state_path="../models/chkp_gnn-model/pytorch_model.bin",
                    group_path="../cryptobench/GCN/test.csv",
                    feature_path="../cryptobench/GCN/test_all_feature.json")
