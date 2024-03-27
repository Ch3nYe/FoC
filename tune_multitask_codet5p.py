import os
import sys
import json
import logging
import warnings
import rouge_score
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    create_model_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether to create model from scratch or not."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code or not."}
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze decoder or not."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )

    def __post_init__(self):
        if self.tokenizer_name == None:
            self.tokenizer_name = self.model_path

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path for train file."},
    )
    cache_train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path for cached train file."},
    )
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path for valid file."},
    )
    cache_valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path for cached valid file."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path for test file."},
    )
    cache_test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path for cached test file."},
    )
    filter_cross_options: Optional[str] = field(
        default=None,
        metadata={"help": ("Filter test set by a cross compilation for fine-tune."
                           "For example, \"gcc-11.2.0_x86_64_O2\", \"gcc-11.2.0_x86_64_*\"")
                },
    )
    source_domain: Optional[str] = field(
        default='pcode',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_domain: Optional[str] = field(
        default='name,brief_comment',
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    max_finetune_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if (
            self.train_file is None
            and self.valid_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.valid_file is not None:
                extension = self.valid_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.cache_train_file is not None:
            os.makedirs(os.path.dirname(self.cache_train_file), exist_ok=True)
        if self.cache_valid_file is not None:
            os.makedirs(os.path.dirname(self.cache_valid_file), exist_ok=True)
        if self.cache_test_file is not None:
            os.makedirs(os.path.dirname(self.cache_test_file), exist_ok=True)


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    do_finetune: Optional[bool] = field(default=False)
    report_to: Optional[str] = field(default='wandb')
    wandb_mode: Optional[str] = field(default="offline")
    wandb_name: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default="huggingface")
    finetune_from_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                'Fine-tune from this path if set. This parameter can only '
                'be set in "fine-tune-only" mode.'
                'If fine-tune the model after training, the base model will be '
                'the last checkpoint in `output_dir`. And the path will be '
                'replaced incorrectly, if this parameter is set.'
            )
        },)
    finetune_num_train_epochs: Optional[int] = field(default=12)
    ddp_timeout: Optional[int] = field(default=7200) # 2h


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def freeze_decoder_except_xattn_codegen(model):
    import numpy as np
    def get_model_size(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_size = sum([np.prod(p.size()) for p in model_parameters])
        return "{}MB".format(round(model_size / 1e+6))
    
    print(f'Para before freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = False

    num_decoder_layers = model.decoder.config.n_layer
    for i in range(num_decoder_layers):
        each_decoder_layer = model.decoder.transformer.h[i]
        if hasattr(each_decoder_layer, 'crossattention'):
            for param in each_decoder_layer.crossattention.parameters():
                param.requires_grad = True
            each_decoder_layer.crossattention.to(torch.float32)

        if hasattr(each_decoder_layer, 'alpha_xattn'):
            each_decoder_layer.alpha_xattn.requires_grad = True
    print(f'Para after freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')

def load_model_and_tokenizer(model_args):
    print(f"[+] loading model from {model_args.model_path}")
    config = AutoConfig.from_pretrained(
        model_args.model_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code
    )
    if model_args.create_model_from_scratch:
        model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_path,
            config=config,
            trust_remote_code=model_args.trust_remote_code,
        )
    if 'codet5p-2b' in model_args.model_path:
        model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
    # check
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    # Frozen part of the model
    if model_args.freeze_decoder:
        print('[+] frozen decoder...')
        if 'codet5p-220m' in model_args.model_path:
            model.decoder.requires_grad_(False)
        elif 'codet5p-2b' in model_args.model_path:
            freeze_decoder_except_xattn_codegen(model)
    print_trainable_parameters(model)
    return model, tokenizer

# PROMPT_DICT
PREFIX_FOR_CODET5P = {
    "name": (
        "recovery function name: "
    ),
    "brief_comment": (
        "summarize in one sentence: "
    ),
    "whole_comment": (
        "summarize in detail: "
    ),
    "brief_comment_and_name": (
        "briefly summarize and recovery function name: "
    )
}

def load_tokenized_data(model_args, data_args, training_args):
    def preprocess_function(examples,prefix,src_domain,tgt_domain,tokenizer):
        source = [prefix + ex for ex in examples[src_domain]]
        target = examples[tgt_domain]
        model_inputs = tokenizer(source, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=data_args.max_target_length, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs
    def preprocess_function_with_decoder_input_ids(examples,prefix,src_domain,tgt_domain,tokenizer):
        tag = "\n\n### Response:"
        tokenized_tag_len = tokenizer(tag,return_tensors='pt').input_ids.shape[1]
        sources = [prefix+ex for ex in examples[src_domain]]
        max_length = data_args.max_source_length-tokenized_tag_len-10
        new_sources = []
        for src in sources:
            tokenized_src = tokenizer(src,return_tensors='pt',max_length=max_length,truncation=True).input_ids
            if tokenized_src.shape[1]==max_length: # too long
                src = tokenizer.batch_decode(tokenized_src, skip_special_tokens=True)[0] + "......"
            new_sources.append(src+tag)
        sources = new_sources
        
        target = [src+tgt+tokenizer.eos_token for src,tgt in zip(sources,examples[tgt_domain])]
        # tokenization
        model_inputs = tokenizer(sources,max_length=data_args.max_source_length, \
                                 padding="max_length",truncation=True)
        labels = tokenizer(target,max_length=data_args.max_source_length+data_args.max_target_length, \
                           padding="max_length", truncation=True)
        # create decoder input
        decoder_input_ids = deepcopy(labels["input_ids"])
        for i in range(len(decoder_input_ids)): # shift
            decoder_input_ids[i] = [tokenizer.eos_token_id] + decoder_input_ids[i][:-1]
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        # create labels
        for attention, line in zip(model_inputs['attention_mask'],labels['input_ids']):
            ignore_len = sum(attention)
            line[:ignore_len] = [-100]*ignore_len
            first_pad = True
            for i in range(ignore_len, len(line)):
                if line[i] == tokenizer.eos_token_id:
                    if first_pad:
                        first_pad = False
                        continue
                    else:
                        line[i] = -100
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    if "codet5p-2b" in model_args.model_path:
        preprocess_func = preprocess_function_with_decoder_input_ids
    elif "codet5p-220m" in model_args.model_path:
        preprocess_func = preprocess_function
    else:
        raise NotImplementedError('[!] unsupported model')
    
    data_files = {}
    if data_args.train_file:
        data_files["train"] = data_args.train_file
    if data_args.valid_file:
        data_files["valid"] = data_args.valid_file
    if data_args.test_file:
        data_files["test"] = data_args.test_file
    raw_datasets = load_dataset('json', data_files=data_files)

    if training_args.do_finetune:
        compiler,arch,bit,opti = data_args.filter_cross_options.split('_')
        filter_func = lambda example: (compiler == '*' or example['compiler'] == compiler) \
                    and (arch == '*' or example['arch'] == arch ) \
                    and (bit == '*' or example['bit'] == bit ) \
                    and (opti == '*' or example['opti'] == opti) 
        raw_datasets['finetune'] = raw_datasets['test'].filter(filter_func)
        raw_datasets['test'] = raw_datasets['test'].filter(lambda example: not filter_func(example))
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name,
                                            use_fast=model_args.use_fast_tokenizer,
                                            trust_remote_code=model_args.trust_remote_code)
    
    tokenized_datasets = defaultdict(list)
    for target_domain in data_args.target_domain:
        print(f"[-] process data on target_domain={target_domain}")
        # filter
        filtered_raw_datasets = raw_datasets.filter(lambda example: example[target_domain] != '')

        # tokenization
        prefix = PREFIX_FOR_CODET5P[target_domain]
        if training_args.do_train:
            train_dataset = filtered_raw_datasets['train']
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))
            train_dataset = train_dataset.map(
                preprocess_func,
                fn_kwargs={
                    "prefix": prefix,
                    "src_domain": data_args.source_domain,
                    "tgt_domain": target_domain,
                    "tokenizer": tokenizer,
                },
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache if training_args.local_rank<=0 else True,
                cache_file_name=data_args.cache_train_file+f".{target_domain}.cache",
                desc="Tokenization on train dataset",
            )
            tokenized_datasets['train'].append(train_dataset)
            
        if training_args.do_finetune:
            finetune_dataset = filtered_raw_datasets['finetune']
            if data_args.max_finetune_samples is not None:
                finetune_dataset = finetune_dataset.select(range(data_args.max_finetune_samples))
            finetune_dataset = finetune_dataset.map(
                preprocess_func,
                fn_kwargs={
                    "prefix": prefix,
                    "src_domain": data_args.source_domain,
                    "tgt_domain": target_domain,
                    "tokenizer": tokenizer,
                },
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=finetune_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache if training_args.local_rank<=0 else True,
                cache_file_name=data_args.cache_test_file+f".finetune.{target_domain}.cache",
                desc="Tokenization on finetune dataset",
            )
            tokenized_datasets['finetune'].append(finetune_dataset)

        if training_args.do_eval:
            valid_dataset = filtered_raw_datasets['valid']
            if data_args.max_valid_samples is not None:
                valid_dataset = valid_dataset.select(range(data_args.max_valid_samples))
            valid_dataset = valid_dataset.map(
                preprocess_func,
                fn_kwargs={
                    "prefix": prefix,
                    "src_domain": data_args.source_domain,
                    "tgt_domain": target_domain,
                    "tokenizer": tokenizer,
                },
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=valid_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache if training_args.local_rank<=0 else True,
                cache_file_name=data_args.cache_valid_file+f".{target_domain}.cache",
                desc="Tokenization on valid dataset",
            )
            tokenized_datasets['valid'].append(valid_dataset)

        if training_args.do_predict or training_args.do_finetune:
            test_dataset = filtered_raw_datasets['test'].shuffle(seed=training_args.seed)
            if data_args.max_test_samples is not None:
                test_dataset = test_dataset.select(range(data_args.max_test_samples))
            test_dataset = test_dataset.map(
                preprocess_func,
                fn_kwargs={
                    "prefix": prefix,
                    "src_domain": data_args.source_domain,
                    "tgt_domain": target_domain,
                    "tokenizer": tokenizer,
                },
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=test_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache if training_args.local_rank<=0 else True,
                cache_file_name=data_args.cache_test_file+f".final.{target_domain}.cache",
                desc="Tokenization on test dataset",
            )
            tokenized_datasets['test'].append(test_dataset)
    
    # merge datasets on different target_domain
    for key in tokenized_datasets.keys():
        _dataset = concatenate_datasets(tokenized_datasets[key])
        if 'test' != key:
            _dataset = _dataset.shuffle(seed=training_args.seed)
        tokenized_datasets[key] = _dataset

    return tokenized_datasets

def change_logging_module(training_args):
    if 'wandb' in training_args.report_to:
        os.environ['WANDB_PROJECT'] = training_args.wandb_project
        os.environ['WANDB_MODE'] = training_args.wandb_mode
        os.environ['WANDB_NAME'] = training_args.wandb_name if training_args.wandb_name else training_args.output_dir

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check args
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    data_args.target_domain = data_args.target_domain.split(',')

    change_logging_module(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    with training_args.main_process_first(desc="[+] main-process first: load data..."):
        tokenized_datasets = load_tokenized_data(model_args, data_args, training_args)
        if training_args.do_train:
            train_dataset_len = len(tokenized_datasets['train'])
        if training_args.do_eval:
            valid_dataset_len = len(tokenized_datasets['valid'])
        if training_args.do_predict or training_args.do_finetune:
            test_dataset_len = len(tokenized_datasets['test'])
        if 'finetune' in tokenized_datasets:
            funetune_dataset_len = len(tokenized_datasets['finetune'])
    # Logging dataset info
    if training_args.local_rank <= 0:
        if 'train' in tokenized_datasets:
            print(f"[-] Sample a data from training set:\n{tokenized_datasets['train'][0]}")
        for key in tokenized_datasets.keys():
            print("[-] tokenized {} set: {}".format(key,len(tokenized_datasets[key])))

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if "codet5p-220m" in model_args.model_path:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    elif "codet5p-2b" in model_args.model_path:
        data_collator = default_data_collator
    else:
        raise NotImplementedError
    
    # Metric
    metric = evaluate.load("rouge")

    # Define the compute_metrics function
    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions

        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        rouge_output = metric.compute(
            predictions=pred_str, references=label_str, use_stemmer=True
            )
        return rouge_output

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'] if training_args.do_train else None,
        eval_dataset=tokenized_datasets['valid'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(os.path.join(training_args.output_dir,"final_checkpoint"))  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else train_dataset_len
        )
        metrics["train_samples"] = min(max_train_samples, train_dataset_len)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    

    if training_args.do_predict:
        print("*** Predict ***")

        predict_results = trainer.predict(tokenized_datasets['test'], metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_test_samples = (
            data_args.max_test_samples if data_args.max_test_samples is not None else test_dataset_len
        )
        metrics["predict_samples"] = min(max_test_samples, test_dataset_len)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                label_ids = predict_results.label_ids
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = tokenizer.batch_decode(
                    label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                prediction_results = [[pred.strip(),label.strip()] for pred,label in zip(predictions,labels)]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
                print("[+] prediction results save to",output_prediction_file)
                with open(output_prediction_file, "w", encoding='utf-8') as f:
                    json.dump(prediction_results,f,indent=4)

    # Fine-tune
    if training_args.do_finetune:
        print("*** Fine-tune ***")
        # get last checkpoint 
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir) # max steps
            
            final_checkpoint = os.path.join(training_args.output_dir,"final_checkpoint")
            if os.path.exists(final_checkpoint):
                last_checkpoint = final_checkpoint
        
        # define checkpoint
        checkpoint = None
        if training_args.finetune_from_model_path is not None:
            checkpoint = training_args.finetune_from_model_path
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            raise ValueError("[!] No checkpoint found for fine-tuning.")
        
        model_args.model_path = checkpoint
        model_args.create_model_from_scratch = False
        model, tokenizer = load_model_and_tokenizer(model_args)

        # re-define training_args
        training_args.do_train = True
        training_args.do_eval = False
        training_args.evaluation_strategy = 'no'
        training_args.num_train_epochs = training_args.finetune_num_train_epochs
        training_args.learning_rate = training_args.learning_rate/2
        training_args.weight_decay = 0
        training_args.warmup_steps = 0
        training_args.output_dir = os.path.join(training_args.output_dir,"tune_test")
        training_args.wandb_mode = 'offline'
        change_logging_module(training_args)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['finetune'] if training_args.do_train else None,
            eval_dataset=tokenized_datasets['valid'] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            # callbacks=[EarlyStoppingCallback(
            #     early_stopping_threshold=0.1,
            #     early_stopping_patience=3
            #     )],
        )
        trainer.train_dataset = tokenized_datasets['finetune']
        train_result = trainer.train()
        trainer.save_model(os.path.join(training_args.output_dir,"final_checkpoint"))

        metrics = train_result.metrics
        metrics["finetune_samples"] = funetune_dataset_len
        trainer.log_metrics("finetune", metrics)
        trainer.save_metrics("finetune", metrics)
        trainer.save_state()

        if training_args.do_predict:
            print("*** Predict using Fine-tuned Model ***")
            predict_results = trainer.predict(tokenized_datasets['test'], metric_key_prefix="predict")
            metrics = predict_results.metrics
            max_test_samples = (
                data_args.max_test_samples if data_args.max_test_samples is not None else test_dataset_len
            )
            metrics["predict_samples"] = min(max_test_samples, test_dataset_len)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = predict_results.predictions
                    label_ids = predict_results.label_ids
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    labels = tokenizer.batch_decode(
                        label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    prediction_results = [[pred.strip(),label.strip()] for pred,label in zip(predictions,labels)]
                    output_prediction_file = os.path.join(training_args.output_dir, "finetuned_generated_predictions.json")
                    print("[+] prediction results save to",output_prediction_file)
                    with open(output_prediction_file, "w", encoding='utf-8') as f:
                        json.dump(prediction_results,f,indent=4)


if __name__ == "__main__":
    main()