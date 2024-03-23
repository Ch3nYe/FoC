# FoC

## Environment Setup

`pip install -r requirements.txt`

## Evaluation

We here provide the evaluation script and test data for FoC-BinLLM in summarizing cryptographic binary functions. 

Please downlaod our model from: google-drive  [link](https://drive.google.com/file/d/1sL0R-xbIYwRfTBPyF5b0WAzs3nqbNp8O/view?usp=sharing)

Run the following cmd:

```bash
CUDA_VISIBLE_DEVICES=0 python ./evaluate_model.py --model_path ./FoC-BinLLM --data_file ./test.json --batch_size 16 --src_domain pcode --tgt_domain comment_and_name --max_tgt_len 256
```

The rouge score will be printed into the command, and if you want to calc other metrics, you can remove the annotation of L215-216 & L268-271 in the `evaluate_model.py` file, **and make sure you have installed their dependence**. 


## Cryptographic Dataset

Big volume, coming soon...