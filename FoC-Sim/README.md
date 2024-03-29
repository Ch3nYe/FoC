# Foc-Sim

## Download model and data

We here provide our model-checkpoint and data for evaluation: google-drive [chkp_gnn-model](https://drive.google.com/file/d/1hktP12SZT2CMiIqcgY_MtILvAKWrxDI8/view?usp=sharing), [GCN-testdata](https://drive.google.com/file/d/1MSLfpof-21p8_xqrID6wwKTIqLjcU92d/view?usp=sharing)

Extract model to `./models/chkp_gnn-model`, and extract data to `./cryptobench/GCN`. 

## Evaluation FoC-Sim

**Step-1** generate embeddings for test set

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_foc_sim.py
```

Please comment `train()` in the `main` module, and uncomment `generate_embedding()`


**Step-2** generate query groups (e.g. XO, XA ...) and calculate similarity

```bash
python3 generate_query_group.py
```

For One-to-One search, set `negative_num = 1`

For One-to-Many search, set `negative_num = 100`

`XO, XC , XA , XM` means different ways of sampling, e.g., if you want to sample across architectures and optimize options, you can set  `XO = True, XC = False, XA = True, XM = False` 

**Step-3** calculate AUC and ROC for One-to-One search

```bash
python3 cal_auc_roc.py
```

**Step-4** calculate recall@k and MRR for One-to-Many search

```bash
python3 cal_recall@k_MRR.py
```