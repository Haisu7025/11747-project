import os
import re
from transformers import AutoTokenizer
import random
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import AdamW, AutoConfig
from modeling import DialogElectraMaskedLM
from tqdm import tqdm, trange
import enchant
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
import collections
import numpy as np
import argparse
import torch
import json
import pickle


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'sep_pos': sep_pos,
                'turn_ids': turn_ids
            }
            for input_ids, input_mask, segment_ids, sep_pos, turn_ids in choices_features
        ]
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop(0)


def get_masked_op(token, tokenizer):
    if np.random.rand() < 0.1:
        return random.choice(list(tokenizer.get_vocab().keys()))
    elif np.random.rand() < 0.2:
        return token
    else:
        return "[MASK]"


def mask_tokens(tokens, aux_tokens=None):
    tokens_ = [t for t in tokens]
    d = enchant.Dict("en_US")
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    
    stem_cnt = collections.defaultdict(int)

    if aux_tokens:
        for token in aux_tokens:
            if token not in stop_words and d.check(token) and token.isalpha() and token not in "MF":
                stem = porter.stem(token)
                stem_cnt[stem] += 1
    
    prob = []
    mapping = {}
    p = 0
    for i, token in enumerate(tokens_):
        if token not in stop_words and d.check(token) and token.isalpha() and token not in "MF":
            stem = porter.stem(token)
            stem_cnt[stem] += 1
            prob.append(stem_cnt[stem] + 1)
            mapping[p] = i
            p += 1
    
    prob = np.array(prob)
    prob = prob / prob.sum()
    for idx in np.random.choice(len(prob), size=int(0.2 * len(prob)), p=prob):
        ridx = mapping[idx]
        # tokens[ridx] = get_masked_op(tokens[ridx], tokenizer)
        tokens_[ridx] = "[MASK]"
    return tokens_


def convert_examples_to_features_mlm(examples, max_seq_length, max_utterance_num,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, position=0, leave=True)):

        choices_features = []
        all_tokens = []

        # for MLM task, text_a is None or a list of question-answer text
        text_a = example.text_a
        text_b = example.text_b

        tokens_b = []
        if not text_b or example.label == 0:
            # print(example.guid)
            # print(example.text_b)
            # print(example.label)
            continue
        for idx, text in enumerate(text_b):
            if len(text.strip()) > 0:
                tokens_b.extend(tokenizer.tokenize(text) + ["[SEP]"])
        
        _truncate_seq_pair([], tokens_b, max_seq_length - 2)

        tokens = ["[CLS]"]
        turn_ids = [0]

        context_len = []
        sep_pos = []
        
        tokens_b_raw = " ".join(tokens_b)
        tokens_b = []
        current_pos = 0
        for toks in tokens_b_raw.split("[SEP]")[-max_utterance_num - 1:-1]:
            context_len.append(len(toks.split()) + 1)
            tokens_b.extend(toks.split())
            tokens_b.extend(["[SEP]"])
            current_pos += context_len[-1]
            turn_ids += [len(sep_pos)] * context_len[-1]
            sep_pos.append(current_pos)
            
        tokens += tokens_b

        segment_ids = [0] * (len(tokens))
        
        sep_pos.append(len(tokens) - 1)

        # perform masking before converting to ids
        old_tokens = [t for t in tokens]
        choices_features = []

        # perform masking
        tokens_a = None
        if text_a is not None:
            tokens_a = tokenizer.tokenize(" ".join(text_a))
        tokens = mask_tokens(tokens, aux_tokens=tokens_a)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        
        # pad label as well
        label = tokenizer.convert_tokens_to_ids(old_tokens)
        label += padding

        input_mask_ = input_mask + padding
        segment_ids_ = segment_ids + padding
        turn_ids_ = turn_ids + padding

        context_len_ = context_len + [-1] * (max_utterance_num - len(context_len))
        sep_pos_ = sep_pos + [0] * (max_utterance_num + 1 - len(sep_pos))

        assert len(sep_pos_) == max_utterance_num + 1
        assert len(input_ids) == max_seq_length
        assert len(input_mask_) == max_seq_length
        assert len(segment_ids_) == max_seq_length
        assert len(context_len_) == max_utterance_num 
        assert len(turn_ids_) == max_seq_length 

        choices_features.append((input_ids, input_mask_, segment_ids_, sep_pos_, turn_ids_))
        all_tokens.append(tokens)

        features.append(
            InputFeatures(
                example_id = example.guid, 
                choices_features = choices_features,
                label = label
                )
        )

    print(f"{len(examples) - len(features)} out of {len(examples)} examples are invalid")
    return features


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_utterance_num",  
                        default=20,
                        type=int,
                        help="The maximum total utterance number.")
    parser.add_argument("--max_grad_norm", 
                        default = 1.0, 
                        type = float,
                        help = "The maximum grad norm for clipping")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--cache_data",
                        default='data/mlm_cache_data.pk',
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=4e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_decouple",
                        default=1,
                        type=int,
                        help="Decoupling Layers.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/electra-large-discriminator")

    # read in training data for MLM
    if os.path.exists(args.cache_data):
        print("use cached feature data ...")
        with open(args.cache_data, 'rb') as f:
            obj = pickle.load(f)
            train_features = obj['train']
            eval_features = obj['dev']
    else:
        with open("data/mlm_corpus.json") as f:
            corpus = json.load(f)
            idx = np.arange(len(corpus))
            np.random.shuffle(idx)
            train_idx = idx[5000:]
            dev_idx = idx[:5000]
        
        train_examples = []
        dev_examples = []
        for i in train_idx:
            c = corpus[i]
            article = c["article"].replace("m :", "M:").replace("f :", "F:")
            article = re.split(r"(f : |m : |M: |F: )", article)
            article = ["".join(i) for i in zip(article[1::2], article[2::2])]
                
            example = InputExample(
                guid=c['id'],
                text_a=c['QA'] if 'QA' in c else None,
                text_b=article,
                label=c['label']
            )
            train_examples.append(example)
        for i in dev_idx:
            c = corpus[i]
            article = c["article"].replace("m :", "M:").replace("f :", "F:")
            article = re.split(r"(f : |m : |M: |F: )", article)
            article = ["".join(i) for i in zip(article[1::2], article[2::2])]
            
            example = InputExample(
                guid=c['id'],
                text_a=c['QA'] if 'QA' in c else None,
                text_b=article,
                label=c['label']
            )
            dev_examples.append(example)

        train_features = convert_examples_to_features_mlm(train_examples, args.max_seq_length, args.max_utterance_num, tokenizer)
        eval_features = convert_examples_to_features_mlm(dev_examples, args.max_seq_length, args.max_utterance_num, tokenizer)

        with open(args.cache_data, 'wb') as f:
            obj = {'train': train_features, 'dev': eval_features}
            pickle.dump(obj, f)

    # training dataset preparation
    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_sep_pos = torch.tensor(select_field(train_features, 'sep_pos'), dtype=torch.long)
    all_turn_ids = torch.tensor(select_field(train_features, 'turn_ids'), dtype = torch.long)
    all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sep_pos, all_turn_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # evaluation dataset preparation
    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_sep_pos = torch.tensor(select_field(eval_features, 'sep_pos'), dtype=torch.long)
    all_turn_ids = torch.tensor(select_field(eval_features, 'turn_ids'), dtype = torch.long)
    all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sep_pos, all_turn_ids, all_label_ids)
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # training
    model_config = AutoConfig.from_pretrained("google/electra-large-discriminator")
    model = DialogElectraMaskedLM(config=model_config)
    model.set_pretrain_electra_weights("google/electra-large-discriminator")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]            
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0

        model.train()
        with tqdm(train_dataloader, desc="Training", position=0, leave=True) as t:
            for step, batch in enumerate(t):
                batch = tuple(tt.to(device) for tt in batch)
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'sep_pos': batch[3],
                            'turn_ids': batch[4],
                            'labels': batch[5]}
                
                output = model(**inputs)
                loss = output[0]
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.detach().item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                t.set_postfix({"loss": tr_loss / (step + 1)})
                t.update()

        model.eval()
        with tqdm(eval_dataloader, desc="Validation", position=0, leave=True) as t:
            for step, batch in enumerate(t):
                batch = tuple(tt.to(device) for tt in batch)
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'sep_pos': batch[3],
                            'turn_ids': batch[4],
                            'labels': batch[5]}
                
                output = model(**inputs)
                loss = output[0]
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.detach().item()

                t.set_postfix({"loss": tr_loss / (step + 1)})
                t.update()
        
        model.save_pretrained(f"output/electra_mlm_{epoch}")


if __name__ == "__main__":
    main()