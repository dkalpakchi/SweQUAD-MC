import os
import sys
import dataclasses as dc

import string

import torch
import torch.nn as nn
import transformers
from transformers import (
    BertConfig,
    AutoTokenizer, AutoModel, BertForMaskedLM,
    HfArgumentParser, TrainingArguments, Trainer,
    DataCollatorForTokenClassification
)
from transformers.integrations import TensorBoardCallback

try:
    from dataset import DistractorDataset, TokenizeTransform, AutoregressiveTransform, UPMLMTransform
    from util import DGArguments
except ModuleNotFoundError:
    from distractors.models.bert_based.dataset import DistractorDataset, TokenizeTransform, AutoregressiveTransform, UPMLMTransform
    from distractors.models.bert_based.util import DGArguments


if __name__ == '__main__':
    import sys
    parser = HfArgumentParser((TrainingArguments, DGArguments))
    train_args, dg_args = parser.parse_args_into_dataclasses()
    print(train_args)
    print(dg_args)

    # transformers.logging.set_verbosity_info()

    tok = AutoTokenizer.from_pretrained(dg_args.model)
    collator = DataCollatorForTokenClassification(tok)

    train_d = DistractorDataset(dg_args.train_data.split(","))
    tokenized_train = TokenizeTransform(train_d, tok)

    if dg_args.formulation == "areg":
        train_ds = AutoregressiveTransform(tokenized_train)
    elif dg_args.formulation == "upmlm":
        train_ds = UPMLMTransform(
            tokenized_train,
            number_of_samples=dg_args.upmlm_samples,
            seed=dg_args.upmlm_seed
        )

    if dg_args.dev_data:
        dev_d = DistractorDataset(dg_args.dev_data.split(","))
        tokenized_dev = TokenizeTransform(dev_d, tok)
        if dg_args.formulation == "areg":
            dev_ds = AutoregressiveTransform(tokenized_dev)
        elif dg_args.formulation == "upmlm":
            dev_ds = UPMLMTransform(
                tokenized_dev,
                number_of_samples=dg_args.upmlm_samples,
                seed=dg_args.upmlm_seed
            )

    model = BertForMaskedLM.from_pretrained(dg_args.model)

    if dg_args.freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if dg_args.dev_data else None,
        data_collator=collator,
        callbacks=[
            TensorBoardCallback()
        ]
    )

    if train_args.do_train:
        torch.save(dg_args, os.path.join(train_args.output_dir, 'dg_args.bin'))
        trainer.train()
        trainer.save_model()