import json
import random
import copy
from itertools import permutations

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def get_text(c, field):
    return c["extra"][field].strip() if "extra" in c and c["extra"] else c["text"].strip()


class GenericDataset(Dataset):
    def __init__(self):
        self._data = []

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class DistractorDataset(GenericDataset):
    def __init__(self, fnames):
        super().__init__()
        for fname in fnames:
            d = json.load(open(fname))
            assert 'data' in d, "Wrong format"
            for dp in d["data"]:
                self._data.append({
                    "context": dp["context"].strip(),
                    "question": dp["question"].strip(),
                    "correct": None,
                    "distractors": set()
                })
                for c in dp["choices"]:
                    if c["type"] == "Correct answer":
                        self._data[-1]["correct"] = get_text(c, "comment")
                    elif c["type"] == "Distractor":
                        self._data[-1]["distractors"].add(get_text(c, "comment"))


class TokenizeTransform(GenericDataset):
    def __init__(self, dataset, tokenizer, max_context_length=384, return_ids=True):
        super().__init__()
        self.__tok = tokenizer
        self.__max_context_length = max_context_length
        self.__return_ids = return_ids
        self.__encode(dataset)

    @property
    def tok(self):
        return self.__tok

    @property
    def max_context_length(self):
        return self.__max_context_length

    def __encode_context(self, ctx, encoder_func):
        if self.__max_context_length:
            return [encoder_func(
                ctx,
                add_special_tokens=False,
                truncation=True,
                max_length=self.__max_context_length
            )]
        else:
            return [encoder_func(ctx, add_special_tokens=False)]


    def __encode(self, dataset):
        for dp in dataset:
            ctx, q = dp["context"], dp["question"]
            a, d = dp["correct"], dp["distractors"]

            if not a.strip() or not q.strip() or any([not dd.strip() for dd in d]):
                continue

            func_name = 'encode' if self.__return_ids else 'tokenize'
            func = getattr(self.__tok, func_name)

            encoded_ctx = self.__encode_context(ctx, func)

            for c in encoded_ctx:
                self._data.append({
                    "context": c,
                    "question": func(q, add_special_tokens=False),
                    "correct": func(a, add_special_tokens=False),
                    "distractors": [
                        func(x, add_special_tokens=False) for x in d
                    ]
                })


class AutoregressiveTransform(GenericDataset):
    def __init__(self, dataset, for_generation=False):
        super().__init__()
        self.__for_generation = for_generation
        self.__encode(dataset)
        
    def __encode(self, dataset):
        def add_datapoints(seq):
            for idx in range(len(seq)):
                if idx > 0:
                    input_ids.append(seq[idx-1])
                    tt_ids.append(1)
                    att_mask.append(1)

                masked_input_ids = list(input_ids)
                masked_input_ids.append(tok.mask_token_id)
                labels = [IGNORE_INDEX] * len(masked_input_ids)
                labels[-1] = seq[idx]
                masked_tt_ids = list(tt_ids)
                masked_tt_ids.append(1)
                masked_att = list(att_mask)
                masked_att.append(1)
                
                self._data.append({
                    "input_ids": masked_input_ids,
                    "token_type_ids": masked_tt_ids,
                    "attention_mask": masked_att,
                    "labels": labels
                })
            else:
                input_ids.append(seq[idx])
                tt_ids.append(1)
                att_mask.append(1)

                masked_input_ids = list(input_ids)
                masked_input_ids.append(tok.mask_token_id)
                labels = [IGNORE_INDEX] * len(masked_input_ids)
                labels[-1] = tok.sep_token_id
                masked_tt_ids = list(tt_ids)
                masked_tt_ids.append(1)
                masked_att = list(att_mask)
                masked_att.append(1)

                self._data.append({
                    "input_ids": masked_input_ids,
                    "token_type_ids": masked_tt_ids,
                    "attention_mask": masked_att,
                    "labels": labels
                })

                input_ids.append(tok.sep_token_id)
                tt_ids.append(1)
                att_mask.append(1)

        assert hasattr(dataset, "tok"), "Run TokenizeTransform first"
        tok = dataset.tok
        for dp in dataset:
            inp = tok.prepare_for_model(
                dp["context"],
                dp["question"]
            )
            inp["input_ids"].extend(dp["correct"])
            inp["input_ids"].append(tok.sep_token_id)
            Nc = len(dp["correct"]) + 1
            part = [1] * Nc
            inp["token_type_ids"].extend(part)
            inp["attention_mask"].extend([1] * Nc)
            input_ids = list(inp["input_ids"])
            tt_ids = list(inp["token_type_ids"])
            att_mask = list(inp["attention_mask"])

            if self.__for_generation:
                input_ids.append(tok.mask_token_id)
                tt_ids.append(1)
                att_mask.append(1)
                
                self._data.append({
                    "input_ids": input_ids,
                    "token_type_ids": tt_ids,
                    "attention_mask": att_mask,
                    "labels": dp["distractors"]
                })
            else:
                for d_tok in dp["distractors"]:
                    add_datapoints(d_tok)


class UPMLMTransform(GenericDataset):
    def __init__(self, dataset, for_generation=False, number_of_samples=20, seed=42):
        super().__init__()
        self.__for_generation = for_generation
        self.__num_samples = number_of_samples
        self.__seed = seed
        self.__encode(dataset)
        
    def __encode(self, dataset):
        if self.__seed:
            random.seed(self.__seed) # for repeatability

        def add_datapoints(seq, prev_seq, correct):
            N = len(seq)

            for _ in range(min(N, self.__num_samples)):
                masked_ratio = random.random()

                masked_input_ids = list(input_ids)
                masked_tt_ids = list(tt_ids)
                masked_att = list(att_mask)
                labels = [IGNORE_INDEX] * len(masked_input_ids)

                for pseq in prev_seq:
                    for p_tok in pseq:
                        masked_input_ids.append(p_tok)
                        masked_tt_ids.append(1)
                        masked_att.append(1)
                        labels.append(IGNORE_INDEX)
                    masked_input_ids.append(tok.sep_token_id)
                    masked_tt_ids.append(1)
                    masked_att.append(1)
                    labels.append(IGNORE_INDEX)

                indeed_masked = 0
                for idx in range(N):
                    is_masked = (N == 1) or random.random() >= masked_ratio
                    
                    if is_masked:
                        indeed_masked += 1
                        masked_input_ids.append(tok.mask_token_id)
                        labels.append(seq[idx])
                    else:
                        masked_input_ids.append(seq[idx])
                        labels.append(IGNORE_INDEX)
                    
                    masked_tt_ids.append(1)
                    masked_att.append(1)

                if indeed_masked == 0: # at least one masked token except [SEP]
                    idx = random.randint(0, N)
                    masked_input_ids[-idx] = tok.mask_token_id
                    labels[-idx] = seq[-idx]

                self._data.append({
                    "input_ids": masked_input_ids,
                    "token_type_ids": masked_tt_ids,
                    "attention_mask": masked_att,
                    "labels": labels
                })

        assert hasattr(dataset, "tok"), "Run TokenizeTransform first"
        tok = dataset.tok
        for dp in dataset:
            inp = tok.prepare_for_model(
                dp["context"],
                dp["question"]
            )
            inp["input_ids"].extend(dp["correct"])
            inp["input_ids"].append(tok.sep_token_id)
            Nc = len(dp["correct"]) + 1 # +1 for [SEP]
            part = [1] * Nc
            inp["token_type_ids"].extend(part)
            inp["attention_mask"].extend([1] * Nc)
            input_ids = list(inp["input_ids"])
            tt_ids = list(inp["token_type_ids"])
            att_mask = list(inp["attention_mask"])

            if self.__for_generation:
                self._data.append({
                    "input_ids": input_ids,
                    "token_type_ids": tt_ids,
                    "attention_mask": att_mask,
                    "labels": dp["distractors"],
                    "ca": dp["correct"]
                })
            else:
                for d_idx, d_tok in enumerate(dp["distractors"]):
                    add_datapoints(d_tok, dp["distractors"][:d_idx], dp["correct"])
