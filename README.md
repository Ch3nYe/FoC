# FoC

## Environment Setup

`pip install -r requirements.txt`

## Evaluation

Downlaod the model from: google-drive link

Run tyhe following cmd:

```bash
CUDA_VISIBLE_DEVICES=0 python ./evaluate_model.py --model_path ./FoC-BinLLM --data_file ./test.json --batch_size 16 --src_domain pcode --tgt_domain comment_and_name --max_tgt_len 256
```

The rouge score will be printed into the command, and if you want to calc other metrics, you can remove the annotation of L215-216 & L268-271 in the `evaluate_model.py` file, and make sure you have installed their dependence. 