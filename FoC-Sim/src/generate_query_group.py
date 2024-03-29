'''
@desc: generate query groups (e.g. XO, XA ...) and calculate similarity
@usage:  python3 generate_query_group.py
'''

import csv
import os
import torch
import pickle
import json
import random
import logging
from tqdm import tqdm
from collections import defaultdict
import pandas as pd 
from torch.nn.functional import cosine_similarity
random.seed(233)

function_groups_path = "../cryptobench/GCN/test.csv"
embedding_path = "../cryptobench/GCN/test.embs.pkl"
dataset_save_to = "./query_group/crypto"

negative_num = 100   # OnetoOne:1  OnetoMany:100

XA = False
XC = False
XO = False
XM = True
DEBUG = False

filename_suffix = "{}{}{}{}{}".format("_balance" if negative_num == 1 else "_unbalance" , "_XC" if XC else "", "_XA" if XA else "", "_XO" if XO else "", "_XM" if XM else "")
dataset_path = dataset_save_to + filename_suffix + ".json"


def create_dataset():

    df_func = pd.read_csv(function_groups_path, index_col=None)
    df_func.reset_index(drop=True, inplace=True)
    func_name_dict = defaultdict(set)
    for i, fname in enumerate(df_func.func_name):
        func_name_dict[fname].add(i)
    func_name_list = list(func_name_dict.keys())
    print("Found {} functions".format(len(func_name_list)))
    random.shuffle(func_name_list)

    if DEBUG:
        func_name_list = func_name_list[:1400]

    query_groups = []

    from copy import deepcopy
    
    # 1. generate query groups
    for fname in tqdm(func_name_list, desc="generate query group"): 
        group = list(func_name_dict[fname])
        random.shuffle(group)
        if len(group) < 2:
            continue
        if XM:
            anchor_id = random.choice(group)
            pos_func_id = random.choice(group)
            while anchor_id == pos_func_id:
                pos_func_id = random.choice(group)
            # sample negative function
            neg_func_ids = []
            _temp_func_name_list = deepcopy(func_name_list)
            random.shuffle(_temp_func_name_list)
            for _fname in _temp_func_name_list:
                if len(neg_func_ids)>=negative_num:
                    break
                if _fname == fname:
                    continue
                _temp = list(func_name_dict[_fname])
                neg_id = random.choice(_temp)
                neg_func_ids.append(neg_id)
            query_groups.append([anchor_id,pos_func_id]+neg_func_ids)

        else:
            anchor_id = group[0]
            pos_flag = False
            for pos_idx in range(len(group)):
                if pos_idx == 0:
                    continue
                if XO and df_func.loc[anchor_id]['opti'] == df_func.loc[group[pos_idx]]['opti'] or not XO and df_func.loc[anchor_id]['opti'] != df_func.loc[group[pos_idx]]['opti']:
                    continue
                if XA and df_func.loc[anchor_id]['arch'] + str(df_func.loc[anchor_id]['bit']) == df_func.loc[group[pos_idx]]['arch'] + str(df_func.loc[group[pos_idx]]['bit']) or not XA and df_func.loc[anchor_id]['arch'] + str(df_func.loc[anchor_id]['bit']) != df_func.loc[group[pos_idx]]['arch'] + str(df_func.loc[group[pos_idx]]['bit']) :
                    continue
                if XC and df_func.loc[anchor_id]['compiler'] == df_func.loc[group[pos_idx]]['compiler'] or not XC and df_func.loc[anchor_id]['compiler'] != df_func.loc[group[pos_idx]]['compiler'] :
                    continue                
                pos_flag = True
                pos_func_id = group[pos_idx]
                break
            
            if pos_flag == False :
                continue
            
            # sample negative function
            neg_func_ids = []
            _temp_func_name_list = deepcopy(func_name_list)
            random.shuffle(_temp_func_name_list)
            for _fname in _temp_func_name_list:
                if len(neg_func_ids)>=negative_num:
                    break
                if _fname == fname:
                    continue
                _temp = list(func_name_dict[_fname])
                neg_flag = False
                for neg_idx in range(len(_temp)):
                    neg_id = _temp[neg_idx]
                    if XO and df_func.loc[anchor_id]['opti'] == df_func.loc[_temp[neg_idx]]['opti'] or not XO and df_func.loc[anchor_id]['opti'] != df_func.loc[_temp[neg_idx]]['opti']:
                        continue
                    if XA and df_func.loc[anchor_id]['arch'] + str(df_func.loc[anchor_id]['bit']) == df_func.loc[_temp[neg_idx]]['arch'] + str(df_func.loc[_temp[neg_idx]]['bit']) or not XA and df_func.loc[anchor_id]['arch'] + str(df_func.loc[anchor_id]['bit']) != df_func.loc[_temp[neg_idx]]['arch'] + str(df_func.loc[_temp[neg_idx]]['bit']) :
                        continue
                    if XC and df_func.loc[anchor_id]['compiler'] == df_func.loc[_temp[neg_idx]]['compiler'] or not XC and df_func.loc[anchor_id]['compiler'] != df_func.loc[_temp[neg_idx]]['compiler'] :
                        continue                
                    neg_flag = True
                    neg_id = _temp[neg_idx]
                    break                    
                if neg_flag:
                    neg_func_ids.append(neg_id)
            
            query_groups.append([anchor_id,pos_func_id]+neg_func_ids)

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(query_groups, f, indent=4)

    # 2. calculate similarity
    with open(embedding_path,"rb") as f:
        embeddings = pickle.load(f)

    save_groups = []
    for group in tqdm(query_groups, desc="calc sim score"):
        anchor_id = group[0]
        anchor_embedding = embeddings[df_func.loc[anchor_id]['fid']]
        pos_func_id = group[1]
        _save_group = []
        for fid in group[1:]:
            fid_embedding = embeddings[df_func.loc[fid]['fid']]
            sim = cosine_similarity(torch.tensor(anchor_embedding),
                                    torch.tensor(fid_embedding),dim=-1)
            _save_group.append(f"{fid}:{sim}")
        _save_group = sorted(_save_group, key=lambda x: float(x.split(":")[-1]), reverse=True)
        _save_group = [(anchor_id,pos_func_id),] + _save_group
        save_groups.append(_save_group)
    save_to = os.path.splitext(dataset_path)[0] + ".scores.json" 
    print(f"[+] similarity results save to {save_to}")

    with open(save_to,'w',encoding="utf-8") as f:
        json.dump(save_groups,f,indent=4)


if __name__ =="__main__":
    create_dataset()