import copy
import sys
import argparse
import os
import string

import torch
import numpy as np
from transformers import (
    AutoTokenizer, BertForMaskedLM, BertConfig
)
from tqdm import tqdm

try:
    from trainer import DGArguments
    from dataset import (
        DistractorDataset, TokenizeTransform, AutoregressiveTransform, UPMLMTransform
    )
except ModuleNotFoundError:
    from distractors.models.bert_based.trainer import DGArguments
    from distractors.models.bert_based.dataset import (
        DistractorDataset, TokenizeTransform, AutoregressiveTransform, UPMLMTransform
    )


class Generator:
    def __init__(self, tokenizer, model, device='cpu', separate_distractor_type=False, force_dg_tokens=False, qad=False):
        self.__tok = tokenizer
        self.__m = model
        self.__device = device
        self.__m.eval()
        self.__m.to(device)

    @property
    def tok(self):
        return self.__tok

    def dict2tensors(self, d):
        for k, v in d.items():
            d[k] = torch.tensor([v]).to(self.__device) # adding batch dimension
        return d
        
    def generate_areg(self, datapoint, num_distractors=3):
        dp = copy.deepcopy(datapoint)
        real_distractors = dp.pop("labels") if "labels" in dp else None

        gen_qa = []
        gen_distractors = []
        inp, out = [], []
        num_gen_distractors = 0
        while num_gen_distractors < num_distractors:
            dis = []

            outputs = self.__m(**self.dict2tensors(dict(dp)))#, output_attentions=True)
            logits = outputs.logits.cpu().detach().numpy()
            tok_id = np.argmax(logits[0][-1]) # get the max logit for the last word
            dis.append(tok_id)
            while tok_id != self.__tok.sep_token_id and len(dis) <= 20:
                dp["input_ids"].insert(-1, tok_id)
                dp["token_type_ids"].append(1)
                dp["attention_mask"].append(1)

                outputs = self.__m(**self.dict2tensors(dict(dp)))
                logits = outputs.logits.cpu().detach().numpy()
                tok_id = np.argmax(logits[0][-1]) # get the max logit for the last word

                dis.append(tok_id)
            dp["input_ids"].insert(-1, tok_id)
            dp["token_type_ids"].append(1)
            dp["attention_mask"].append(1)

            if dis[-1] != self.__tok.sep_token_id:
                dp["input_ids"].insert(-1, self.__tok.sep_token_id)
                dp["token_type_ids"].append(1)
                dp["attention_mask"].append(1)   

            gen_distractors.append(dis)
            dis = []
            num_gen_distractors += 1

        return {
            "distractors": {
                "g": [self.__tok.decode(d) for d in gen_distractors],
                "r": [self.__tok.decode(d) for d in real_distractors] if real_distractors else None
            },
            "inputs": inp,
            "outputs": out
        }


    def generate_upmlm(self, datapoint, num_distractors=3, non_linear_order=False, length_strategy='min_first'):
        def add_masked(dx, length):
            for x in range(length):
                dx["input_ids"].append(self.__tok.mask_token_id)
                dx["token_type_ids"].append(1)
                dx["attention_mask"].append(1)
            return dx

        def replace_masked(dx, lst):
            rep = 0
            for i, x in enumerate(dx["input_ids"]):
                if x == self.__tok.mask_token_id:
                    dx["input_ids"][i] = lst[rep]
                    rep += 1
            return dx

        dp = copy.deepcopy(datapoint) # not to mess up the original dataset

        real_distractors = dp.pop("labels") if "labels" in dp else None
        ca = dp.pop("ca") if "ca" in dp else None

        # Generate distractors
        if real_distractors and ca:
            dis_lengths = [len(x) for x in real_distractors]
            dis_lengths.append(len(ca))

            Ndl = len(dis_lengths)
            if Ndl < num_distractors:
                mnd, mxd = min(dis_lengths), max(dis_lengths)
                for i in range(num_distractors - Ndl):
                    dis_lengths.append(np.random.randint(mnd, mxd + 1))
        elif ca:
            Nc = len(ca)
            dis_lengths.append(Nc)
            for x in range(1,num_distractors // 2):
                dis_lengths.append(Nc - x)
                if len(dis_lengths) < num_distractors:
                    dis_lengths.append(Nc + x)
        else:
            dis_lengths = np.random.randint(0, 20, size=(num_distractors,))

        if length_strategy == 'min_first':
            dis_lengths.sort()
        elif length_strategy == 'max_first':
            dis_lengths.sort()
            dis_lengths.reverse()
        elif length_strategy == 'random':
            np.random.seed(42)
            np.random.shuffle(dis_lengths)

        gen_distractors = []
        inp, out = [], []
        num_gen_distractors = 0 
        while num_gen_distractors < num_distractors:
            dp = add_masked(dp, dis_lengths[num_gen_distractors])

            if non_linear_order:
                dis = [-1] * dis_lengths[num_gen_distractors]
                mask = [False] * dis_lengths[num_gen_distractors]
                while not all(mask):
                    outputs = self.__m(**self.dict2tensors(dict(dp)))
                    logits = outputs.logits.cpu().detach().numpy()
                    dis_logits = logits[0][-dis_lengths[num_gen_distractors]:]
                    dis_logits[mask] = -100
                    d_pos, tok_id = np.unravel_index(np.argmax(dis_logits), dis_logits.shape) # get the max logit for the whole distractor

                    dp["input_ids"][-dis_lengths[num_gen_distractors] + d_pos] = tok_id
                    dis[d_pos] = tok_id
                    mask[d_pos] = True
            else:
                outputs = self.__m(**self.dict2tensors(dict(dp)))
                logits = outputs.logits.cpu().detach().numpy()
                dis = np.argmax(logits[0][-dis_lengths[num_gen_distractors]:], axis=1)

                dp = replace_masked(dp, dis)

            if dis[-1] != self.__tok.sep_token_id:
                dp["input_ids"].append(self.__tok.sep_token_id)
                dp["token_type_ids"].append(1)
                dp["attention_mask"].append(1)

            gen_distractors.append(dis)
            num_gen_distractors += 1

        return {
            "distractors": {
                "g": [self.__tok.decode(d) for d in gen_distractors],
                "r": [self.__tok.decode(d) for d in real_distractors] if real_distractors else None
                # 'gt': gen_distractors,
                # 'rt': real_distractors
            },
            "inputs": inp,
            "outputs": out
        }

    def generate(self, datapoint, num_distractors=3, formulation='areg', non_linear_order=False, length_strategy='min_first'):
        if formulation == 'areg':
            return self.generate_areg(datapoint, num_distractors)
        elif formulation == 'upmlm':
            return self.generate_upmlm(datapoint, num_distractors, non_linear_order=non_linear_order, length_strategy=length_strategy)
        else:
            raise NotImplementedError("The formulation {} is not implemented!".format(formulation))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="Checkpoint path")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset path")
    parser.add_argument('-dg', '--dg-file', type=str, required=True, help="DG settings file")
    parser.add_argument('-o', '--output', type=str, help="Output file")
    parser.add_argument('-fdgt', '--force-dg-tokens', action='store_true', help='Force adding distractor tokens when generating')
    parser.add_argument('-nlo', '--non-linear-order', action='store_true', help='A flag to generate in a non-linear order (works only for u-PMLM models)')
    parser.add_argument('-n', '--num-distractors', default=3, type=int, help='Number of didstractors to be generated')
    parser.add_argument('-ls', '--length-strategy', default='min_first', help='The order of generation for u-PMLM models: one of min_first, max_first or random')
    args = parser.parse_args()

    path = args.file
    dg_args = torch.load(args.dg_file)

    if args.output:
        out_file = open(args.output, 'w')
        sys.stdout = out_file

    tok = AutoTokenizer.from_pretrained(dg_args.model, local_files_only=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
    
    gen = Generator(tok, model, device)

    dev_d = DistractorDataset([args.dataset])
    tokenized_dev = TokenizeTransform(dev_d, tok)

    if dg_args.formulation == 'areg':
        dev_ds = AutoregressiveTransform(tokenized_dev, for_generation=True)
    elif dg_args.formulation == 'upmlm':
        dev_ds = UPMLMTransform(tokenized_dev, for_generation=True)

    for dp in dev_ds:
        print(tok.decode(dp["input_ids"]))
        res = gen.generate(
            dp, formulation=dg_args.formulation,
            non_linear_order=args.non_linear_order,
            num_distractors=args.num_distractors,
            length_strategy=args.length_strategy
        )
        print(res["distractors"]["g"])
        print(res["distractors"]["r"])
        print()

    if args.output:
        out_file.close()