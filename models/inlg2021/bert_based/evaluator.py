import json
import re
import argparse
from difflib import SequenceMatcher
from pprint import pprint
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from terminaltables import AsciiTable

from transformers import AutoTokenizer

import stanza

import udon2
from udon2.kernels import ConvPartialTreeKernel


def exists_in_distractors(distractors, dataset):
    data = dataset["data"]
    for x in data:
        for alt in x["choices"]:
            comment = alt["extra"].get("comment") if alt["extra"] else None
            if alt["type"] == "Distractor" and (alt["text"] in distractors or (comment and comment in distractors)):
                return True
    return False


def all_exist_in_distractors(distractors, dataset):
    data = dataset["data"]
    mask = [False] * len(distractors)
    for x in data:
        for alt in x["choices"]:
            comment = alt["extra"].get("comment") if alt["extra"] else None
            if alt["type"] == "Distractor":
                for i, d in enumerate(distractors):
                    if alt["text"] == d or (comment and comment == d):
                        mask[i] = True
    return all(mask)


def exists_in_context(distractors, dataset):
    if type(dataset) == str:
        for d in distractors:
            if dataset.find(d) != -1:
                return True
    else:
        data = dataset["data"]
        for x in data:
            for d in distractors:
                if x["context"].find(d) != -1:
                    return True
    return False


def all_exist_in_context(distractors, dataset):
    mask = [False] * len(distractors)
    if type(dataset) == str:
        for i, d in enumerate(distractors):
            if dataset.find(d) != -1:
                mask[i] = True
    else:
        data = dataset["data"]
        for x in data:
            for i, d in enumerate(distractors):
                if x["context"].find(d) != -1:
                    mask[i] = True
    return all(mask)


def is_same_context(ctx, dataset, overlap=False):
    if overlap:
        Nctx = len(ctx)
        limit = Nctx / 4
        data = dataset["data"]
        for x in data:
            match = SequenceMatcher(None, x["context"], ctx).find_longest_match(0, len(x["context"]), 0, Nctx)
            if match.size > limit:
                return True
    else:
        data = dataset["data"]
        for x in data:
            if x["context"] == ctx:
                return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="Report file to process")
    parser.add_argument('-t', '--training-data', type=str, default="", help="Training data file")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")

    training_data = json.load(open(args.training_data)) if args.training_data else None

    SPECIAL_TOKENS_REGEX = r"(\[SEP\]|\[[A-Z]\]|')"

    sv = stanza.Pipeline(lang="sv", processors='tokenize,lemma,pos,depparse')
    so_kernel = ConvPartialTreeKernel("GRCT", includeForm=False)
    so_feats_kernel = ConvPartialTreeKernel("GRCT", includeForm=False, includeFeats=True)

    examples = []
    report = {
        "total": 0,
        "correct_in_distractors": [],
        "any_same_distractors": [],
        "all_same_distractors": [],
        "avg_length_difference": defaultdict(list),
        "any_different_capitalization": [],
        "any_start_with_same_word": [],
        "all_start_with_same_word": [],
        "subseq_repetitive_words": [],
        "empty_distractors": [],
        "any_exists_in_context": [],
        "any_exists_in_context_and_training_ctx": [],
        "any_exists_in_context_and_training_dis": [],
        "any_exists_in_training_distractors": [],
        "any_exists_in_training_context": [],
        "all_exist_in_context": [],
        "all_exist_in_context_and_training_ctx": [],
        "all_exist_in_context_and_training_dis": [],
        "all_exist_in_training_distractors": [],
        "all_exist_in_training_context": [],
        "is_context_in_training_data": [],
        "context_overlaps_with_training_data": [],
        "any_predicted_gold_distractors": [],
        "ca_norm_tree_kernel": [],
        "ca_feats_norm_tree_kernel": [],
        "start_with_same_pos": 0,
        "start_with_same_pos_morph": 0,
        "tp": 0,
        "p": 0
    }
    inside_example, current_id = False, -1

    gen_context_position = []

    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[CLS]"):
                inside_example = True
                correct = re.sub(SPECIAL_TOKENS_REGEX, "", line.split("[SEP]")[2])
                examples.append({
                    "text": line,
                    "correct": correct.strip(),
                    "gold": None,
                    "gen": None
                })
                current_id += 1
            elif inside_example:
                if examples[-1]["gen"]:
                    examples[-1]["gold"] = [
                        re.sub(SPECIAL_TOKENS_REGEX, "", x).strip()
                        for x in line[1:-1].split("', '")
                    ]
                else:
                    examples[-1]["gen"] = [
                        re.sub(SPECIAL_TOKENS_REGEX, "", x).strip()
                        for x in line[1:-1].split("', '")
                    ]

                gen, gold = examples[-1]["gen"], examples[-1]["gold"]
                correct = examples[-1]["correct"]
                context = examples[-1]["text"]

                if gen and gold:
                    set_gen, set_gold = set(gen), set(gold)

                    ca = udon2.Importer.from_stanza(sv(correct).to_dict())[0]
                    ca_norm = np.sqrt(so_kernel(ca, ca))
                    ca_feats_norm = np.sqrt(so_feats_kernel(ca, ca))
                    for g in set_gen:
                        if g:
                            gd = udon2.Importer.from_stanza(sv(g).to_dict())[0]
                            report["ca_norm_tree_kernel"].append(
                                so_kernel(ca, gd) / (ca_norm * np.sqrt(so_kernel(gd, gd))))
                            report["ca_feats_norm_tree_kernel"].append(
                                so_feats_kernel(ca, gd) / (ca_feats_norm * np.sqrt(so_feats_kernel(gd, gd))))
                            report["tp"] += g in set_gold
                    report["p"] += len(set_gold)

                    if correct in set_gen:
                        report["correct_in_distractors"].append(current_id)
                    if len(set_gen) != len(gen):
                        report["any_same_distractors"].append(current_id)
                        if len(set_gen) == 1:
                            report["all_same_distractors"].append(current_id)

                    if set_gen & set_gold:
                        report["any_predicted_gold_distractors"].append(current_id)

                    correct_capitalized = correct[0].isupper()
                    if any([correct_capitalized != x[0].isupper() for x in gen if x]):
                        report["any_different_capitalization"].append(current_id)

                    # this assumes a whitespace tokenization
                    cwords = correct.split()
                    dwords = [x.split() for x in gen]
                    Nc, Nd = len(cwords), [len(x) for x in dwords]
                    diff = [abs(Nc - Ndd) for Ndd in Nd]
                    report["avg_length_difference"][sum(diff) / len(diff)].append(current_id)

                    if any([x == 0 for x in Nd]):
                        report["empty_distractors"].append(current_id)

                    same_first_word = [cwords[0] == x[0] for x in dwords if x]

                    all_same_first_word = all(same_first_word)
                    if any(same_first_word) and not all_same_first_word:
                        report["any_start_with_same_word"].append(current_id)
                    if all_same_first_word:
                        report["all_start_with_same_word"].append(current_id)

                    if any([any([y == z for y, z in zip(x[:-1], x[1:])]) for x in dwords]):
                        report["subseq_repetitive_words"].append(current_id)

                    inside_example = False
                    report["total"] += 1

                    if is_same_context(context, training_data):
                        report["is_context_in_training_data"].append(current_id)

                    # if is_same_context(context, training_data, overlap=True):
                    #     report["context_overlaps_with_training_data"].append(current_id)

                    if training_data:
                        gen_in_train_ctx = exists_in_context(gen, training_data)
                        gen_in_train_dis = exists_in_distractors(gen, training_data)

                        if gen_in_train_ctx:
                            all_gen_in_train_ctx = all_exist_in_context(gen, training_data)
                        else:
                            all_gen_in_train_ctx = False

                        if gen_in_train_dis:
                            all_gen_in_train_dis = all_exist_in_distractors(gen, training_data)
                        else:
                            all_gen_in_train_dis = False
                    else:
                        gen_in_train_ctx, gen_in_train_dis = False, False

                    if exists_in_context(gen, context):
                        report["any_exists_in_context"].append(current_id)

                        if all_exist_in_context(gen, context):
                            report["all_exist_in_context"].append(current_id)

                            if all_gen_in_train_ctx:
                                report["all_exist_in_context_and_training_ctx"].append(current_id)

                            if all_gen_in_train_dis:
                                report["all_exist_in_context_and_training_dis"].append(current_id)

                        if gen_in_train_ctx:
                            report["any_exists_in_context_and_training_ctx"].append(current_id)

                        if gen_in_train_dis:
                            report["any_exists_in_context_and_training_dis"].append(current_id)

                    
                    if gen_in_train_ctx:
                        report["any_exists_in_training_context"].append(current_id)

                        if all_gen_in_train_ctx:
                            report["all_exist_in_training_context"].append(current_id)

                    if gen_in_train_dis:
                        report["any_exists_in_training_distractors"].append(current_id)

                        if all_gen_in_train_dis:
                            report["all_exist_in_training_distractors"].append(current_id)

                    for gdis in gen:
                        ddp = context.find(gdis)
                        if ddp > -1:
                            gen_context_position.append(len(tok.tokenize(context[:ddp])))
                        
            else:
                inside_example = False

    # pprint(report)

    print(len(report["ca_norm_tree_kernel"]))
    mode = stats.mode(report["ca_norm_tree_kernel"])
    feats_mode = stats.mode(report["ca_feats_norm_tree_kernel"])

    table_data = [
        ["Metric", "Value"],
        ["Total", report["total"]],
        ["Any of the generated distractors matches with a gold one", "{}%".format(
            round(len(report["any_predicted_gold_distractors"]) * 100 / report["total"], 2))],
        ["The correct answer is among generated distractors", "{}%".format(
            round(len(report["correct_in_distractors"]) * 100 / report["total"], 2))],
        ["Any (but not all) generated distractors are the same", "{}%".format(
            round(len(report["any_same_distractors"]) * 100 / report["total"], 2))],
        ["All generated distractors are the same", "{}%".format(
            round(len(report["all_same_distractors"]) * 100 / report["total"], 2))],
        ["Any distractor is capitalized differently from the correct answer", "{}%".format(
            round(len(report["any_different_capitalization"]) * 100 / report["total"], 2))],
        ["Any distractor contains repetitive words", "{}%".format(
            round(len(report["subseq_repetitive_words"]) * 100 / report["total"], 2))],
        ["Any distractor is an empty string", "{}%".format(
            round(len(report["empty_distractors"]) * 100 / report["total"], 2))],
        ["(A) Any distractor is in its own context", "{}%".format(
            round(len(report["any_exists_in_context"]) * 100 / report["total"], 2))],
        ["(B) Any distractor is in any context from training data", "{}%".format(
            round(len(report["any_exists_in_training_context"]) * 100 / report["total"], 2))],
        ["(C) Any distractor is a distractor from training data", "{}%".format(
            round(len(report["any_exists_in_training_distractors"]) * 100 / report["total"], 2))],
        ["(A) and (B)", "{}%".format(
            round(len(report["any_exists_in_context_and_training_ctx"]) * 100 / report["total"], 2))],
        ["(A) and (C)", "{}%".format(
            round(len(report["any_exists_in_context_and_training_dis"]) * 100 / report["total"], 2))],
        ["(A1) All distractors are in their own context", "{}%".format(
            round(len(report["all_exist_in_context"]) * 100 / report["total"], 2))],
        ["(B1) All distractors are in any context from training data", "{}%".format(
            round(len(report["all_exist_in_training_context"]) * 100 / report["total"], 2))],
        ["(C1) All distractors are distractors from training data", "{}%".format(
            round(len(report["all_exist_in_training_distractors"]) * 100 / report["total"], 2))],
        ["(A1) and (B1)", "{}%".format(
            round(len(report["all_exist_in_context_and_training_ctx"]) * 100 / report["total"], 2))],
        ["(A1) and (C1)", "{}%".format(
            round(len(report["all_exist_in_context_and_training_dis"]) * 100 / report["total"], 2))],
        ["Normalized conv. kernel (SO)", "{} +/- {}".format(
            round(np.mean(report["ca_norm_tree_kernel"]), 2),
            round(np.std(report["ca_norm_tree_kernel"]), 2))],
        ["Median normalized conv. kernel (SO)", "{}".format(
            round(np.median(report["ca_norm_tree_kernel"]), 2))],
        ["Mode normalized conv. kernel (SO)", "{} ({}%)".format(
            round(mode[0][0], 2), round(mode[1][0] * 100 / len(report["ca_norm_tree_kernel"]), 2))],
        ["Normalized conv. kernel (SO, feats)", "{} +/- {}".format(
            round(np.mean(report["ca_feats_norm_tree_kernel"]), 2),
            round(np.std(report["ca_feats_norm_tree_kernel"]), 2))],
        ["Median normalized conv. kernel (SO, feats)", "{}".format(
            round(np.median(report["ca_feats_norm_tree_kernel"]), 2))],
        ["Mode normalized conv. kernel (SO, feats)", "{} ({}%)".format(
            round(feats_mode[0][0], 2), round(feats_mode[1][0] * 100 / len(report["ca_feats_norm_tree_kernel"]), 2))],
        ["Distractor recall", "{}%".format(round(report["tp"] * 100 / report["p"], 2))]
        # ["A context exists in training data", "{}%".format(
        #     round(len(report["is_context_in_training_data"]) * 100 / report["total"], 2))]
        # ["A context overlaps with training data (> 25\% overlap)", "{}%".format(
        #     round(len(report["context_overlaps_with_training_data"]) * 100 / report["total"], 2))]
    ]

    t = AsciiTable(table_data)
    print(t.table)

    plt.hist(gen_context_position, bins=range(min(gen_context_position), max(gen_context_position)))
    plt.show()