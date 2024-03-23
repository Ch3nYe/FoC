'''
@description: evaluate model performance
@usage: 
CUDA_VISIBLE_DEVICES=0 python ./evaluate_model.py --model_path ./FoC-BinLLM --data_file ./test.json --batch_size 16 --src_domain pcode --tgt_domain comment_and_name --max_tgt_len 256
'''
import os
import time
import json
import torch
import evaluate
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from typing import Optional, Union, List, Dict, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, 
    HfArgumentParser, DataCollatorForSeq2Seq, 
)
from accelerate import Accelerator


@dataclass
class DataArguments:
    model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model"},
    )
    src_domain: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    tgt_domain: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    filter_empty_tgt_domain: Optional[bool] = field(
        default=True,
        metadata={"help": "not filter empty tgt domain, default False."},
    )
    data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json file)."}
    )
    max_data_num: Optional[int] = field(
        default=None, metadata={"help": "The max data number."}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "A path store the whole of dataset."
            )
        },
    )
    max_src_len: Optional[int] = field(
        default=1024,
        metadata={"help": "max length of source"}
    )
    max_tgt_len: Optional[int] = field(
        default=32,
        metadata={"help": "max length of target"}
    )
    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "batch size per device"}
    )

    def __post_init__(self):
        if (
            self.model_path is None
            or self.src_domain is None
            or self.tgt_domain is None
        ):
            raise Exception('args error')
        elif self.data_file is None and self.data_path is None:
            raise Exception('args error')
    

def load_tokenize_data(args, accelerator):
    def preprocess_function_name(examples):
        prefix = "recovery function name:"
        src_domain = 'pcode'
        tgt_domain = 'name'
        source,target = [],[]
        if 'strip_name' not in examples:
            for ex,name in zip(examples[src_domain],examples[tgt_domain]):
                if "<FUNCTION>" not in ex:
                    ex = ex.replace(name,"<FUNCTION>")
                ex = prefix + ex
                source.append(ex)
                target.append(name)
        else:
            for ex,name,strip_name in zip(examples[src_domain],examples[tgt_domain],examples['strip_name']):
                if "<FUNCTION>" not in ex:
                    ex = ex.replace(name,"<FUNCTION>").replace(strip_name,"<FUNCTION>")
                ex = prefix + ex
                source.append(ex)
                target.append(name)

        model_inputs = tokenizer(source, max_length=args.max_src_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_tgt_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs
    
    def preprocess_function_comment(examples):
        prefix = "summarize in one sentence:"
        src_domain = 'pcode'
        tgt_domain = 'comment'

        source = [prefix + ex for ex in examples[src_domain]]
        target = examples[tgt_domain]

        model_inputs = tokenizer(source, max_length=args.max_src_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_tgt_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs
    
    def preprocess_function_comment_and_name(examples):
        prefix = ""
        src_domain = 'pcode'
        tgt_domain = 'comment_and_name'

        source = [prefix + ex for ex in examples[src_domain]]
        target = examples[tgt_domain]

        model_inputs = tokenizer(source, max_length=args.max_src_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_tgt_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs
    if args.tgt_domain == "comment_and_name":
        preprocess_function = preprocess_function_comment_and_name
        filter_func = lambda example: example['comment_and_name'] != ""
    # elif args.tgt_domain == "name":
    #     preprocess_function = preprocess_function_name
    #     filter_func = lambda example: example['name'] != ""
    # elif args.tgt_domain == "wizardcoder_comment":
    #     preprocess_function = preprocess_function_comment
    #     filter_func = lambda example: example['comment'] != ""
    else:
        raise NotImplementedError
    
    print("[-] loading dataset...")
    if args.data_file is not None:
        raw_datasets = load_dataset("json", data_files={'train':args.data_file})
        if args.max_data_num:
            raw_datasets['train'] = raw_datasets['train'].select(range(args.max_data_num))
        if args.filter_empty_tgt_domain:
            raw_datasets['train'] = raw_datasets['train'].filter(filter_func)
        remove_columns = raw_datasets['train'].column_names
        if 'fid' in remove_columns:
            remove_columns.remove('fid')
        with accelerator.main_process_first():
            dataset = raw_datasets['train'].map(
                preprocess_function,
                batched=True,
                remove_columns=remove_columns,
                desc="Running tokenizer on dataset",
            )
    else:
        raise NotImplementedError
    print("[-] data count:",len(dataset))
    return dataset

def main():
    parser = HfArgumentParser(DataArguments)
    args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator()
    # accelerator.mixed_precision == 'fp16'
    checkpoint_path = args.model_path

    print("load model from:",checkpoint_path)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    config.hidden_size = config.d_model
    if config.decoder_start_token_id is None:
        config.decoder_start_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.unk_token_id = tokenizer.unk_token_id
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, config=config, 
                                                    torch_dtype=torch.float16, 
                                                    trust_remote_code=True)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    valid_dataset = load_tokenize_data(args, accelerator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    model = accelerator.prepare(model)
    valid_dataloader = accelerator.prepare(valid_dataloader)
    predctions = []
    targets = []
    model.eval()
    print("load metrics...")
    rouge = evaluate.load("rouge")
    # bleu = evaluate.load("bleu")
    # meteor = evaluate.load("meteor")
    print("start evaluating...")
    for batch in tqdm(valid_dataloader):
        labels = batch.pop("labels")
        fids = batch.pop("fid") if "fid" in batch else torch.zeros(labels.shape[0],device=labels.device)
        with torch.no_grad():
            outputs = model(**batch)
            outputs = outputs.logits.argmax(dim=-1)
        # Gather all predictions and targets
        outputs, labels, fids = accelerator.gather_for_metrics((outputs, labels, fids))
        fids = fids.cpu().tolist()
        pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predctions.extend([{str(fid):p} for p,fid in zip(pred_str,fids)])
        targets.extend([{str(fid):l} for l,fid in zip(label_str,fids)])
        print("="*80,'\n',predctions[-1],'\n',targets[-1]) # debug
        if accelerator.is_main_process:
            try:
                rouge_output = rouge.compute(predictions=pred_str, references=label_str)
                print("[-] batch metric:\n",rouge_output)
            except:
                print("[!] get rouge scores error!")

    if accelerator.is_main_process:
        # save predictions and references to file
        timestap = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        save_to = os.path.join(checkpoint_path,f"{__file__}.{os.path.basename(args.data_file)}.{args.src_domain}2{args.tgt_domain}.{timestap}.json")
        pred_results = dict()
        for p_item,t_item in zip(predctions,targets):
            idx = list(p_item.keys())[0]
            pred_results[idx] = [p_item[idx],t_item[idx]]
        print("[+] save results to:",save_to)
        with open(save_to,"w") as f:
            json.dump(pred_results,f,indent=4)
        
        # calc final metrics
        predctions = [list(item.values())[0] for item in predctions]
        targets = [list(item.values())[0] for item in targets]
        def get_comment(s):
            s = s.split('\n')[0]
            s = s.replace("<COMMENT>","")
            s = s.replace("</COMMENT>","")
            return s
        preds = []
        for p in predctions:
            preds.append(get_comment(p))
        refs = []
        for t in targets:
            refs.append(get_comment(t))
        rouge_scores = rouge.compute(predictions=preds, references=refs)
        print(f"[+] binary code summarization scores:\n",rouge_scores)
        # bleu_scores = bleu.compute(predictions=preds, references=refs, max_order=4, smooth=False)
        # print("bleu scores:\n",bleu_scores)
        # meteor_scores = meteor.compute(predictions=preds, references=refs)
        # print("meteor scores:\n",meteor_scores)

if __name__ == "__main__":
    main()
