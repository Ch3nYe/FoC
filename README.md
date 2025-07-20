<div align="center">
<h1>FoC: Figure out the Cryptographic Functions in Stripped Binaries with LLMs</h1>
<p>
<a href="https://dl.acm.org/doi/10.1145/3731449">Paper Link: https://dl.acm.org/doi/10.1145/3731449</a>
</p>
</div>

## Environment Setup

`pip install -r requirements.txt`

## Evaluation

We here provide the evaluation script and the *x86_64* test data (`test.json`) for FoC-BinLLM in summarizing cryptographic binary functions. 

Please downlaod our model from: google-drive  [link](https://drive.google.com/file/d/1sL0R-xbIYwRfTBPyF5b0WAzs3nqbNp8O/view?usp=sharing)

Run the following cmd:

```bash
CUDA_VISIBLE_DEVICES=0 python ./evaluate_model.py --model_path ./FoC-BinLLM --data_file ./test.json --batch_size 16 --src_domain pcode --tgt_domain comment_and_name --max_tgt_len 256
```

The rouge score will be printed into the command, and if you want to calc other metrics, you can remove the annotation of L215-216 & L268-271 in the `evaluate_model.py` file, **and make sure you have installed their dependence**. 


## Training

We provide the training script for FoC-BinLLM, you can use the following bash script to train the model:

`./tune_all-multitask_codet5p-220m.sh`

The pre-processed dataset needed when training could be found in google-drive: [link](https://drive.google.com/file/d/1GkxjL8NZb4heCjmJXx9e9qnBmKLMHJRP/view?usp=sharing)

Extract the data file and put them into the `datasets/all-multitask` directory before training. 

The binaries (big volume) will be released soon ...

## FoC-Sim

The second module of FoC could be found in the `FoC-Sim` directory. 

## Citation

If you feel our work insightful, please consider cite this:

```
@article{10.1145/3731449,
author = {Shang, Xiuwei and Chen, Guoqiang and Cheng, Shaoyin and Guo, Shikai and Zhang, Yanming and Zhang, Weiming and Yu, Nenghai},
title = {FoC: Figure out the Cryptographic Functions in Stripped Binaries with LLMs},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1049-331X},
url = {https://doi.org/10.1145/3731449},
doi = {10.1145/3731449},
journal = {ACM Trans. Softw. Eng. Methodol.},
month = apr,
keywords = {Binary Code Summarization, Cryptographic Algorithm Identification, Binary Code Similarity Detection, Large Language Models}
}
```
