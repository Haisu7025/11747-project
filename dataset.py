# -*- coding: utf-8 -*-
"""“dataset.ipynb”的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_DzDMle9xX6t94rXnwl7iuizjQOZSQQ9

# Process Dialogue Corpus
"""

import os
import re
import glob
import json
from tqdm import tqdm

import numpy as np
from google.colab import drive
drive.mount('/content/drive')

"""## Process DREAM"""

with open("/content/drive/MyDrive/datasets/DREAM/train.json") as f:
    dream_train = json.load(f)
# with open("/content/drive/MyDrive/datasets/DREAM/dev.json") as f:
#     dream_dev = json.load(f)
# with open("/content/drive/MyDrive/datasets/DREAM/test.json") as f:
#     dream_test = json.load(f)

len(dream_train)

"""## Process Topical-chat"""

with open("/content/drive/MyDrive/datasets/TOPICAL-CHAT/train.json") as f:
    tc = json.load(f)

# topical_train = []
# for conversation_id, conversation_dict in tc.items():
#     conversation = []
#     for turn in conversation_dict['content']:
#       conversation.append(turn['agent'].replace('agent_1', 'M').replace('agent_2', 'F') + ': ' + turn['message'][0])
#     topical_train.append(conversation)


# def truncate_utterance(article):
#     if len(article) <= 2:
#         return None
#     new_article_len = np.random.randint(len(article) // 2, len(article))
#     start_idx = np.random.randint(0, len(article) - new_article_len)
#     end_idx = start_idx + new_article_len
#     new_article = article[start_idx: end_idx]
#     return new_article

"""## Process Mutual"""

mutual = []
with open("/content/drive/MyDrive/datasets/MUTUAL/mutual_train.json", 'r') as f:
    for line in f.readlines():
        mutual.append(json.loads(line))

len(mutual)

"""# Build training corpus"""

corpus = []
positive_corpus = []
negative_corpus = []

label_list = ["A", "B", "C", "D"]
for example in mutual:
    for i, option in enumerate(example['options']):
        label = label_list.index(example['answers'])
        if label == i:
          obj = {
              "id": f"mutual-{example['id']}-{i}",
              "article": (example['article']).replace("m :", "M:").replace("f :", "F:"),
              "last-sentence": option.replace("m :", "M:").replace("f :", "F:"),
              "label": 1
          }
          positive_corpus.append(obj)
        else:
          obj = {
              "id": f"mutual-{example['id']}-{i}",
              "article": (example['article']).replace("m :", "M:").replace("f :", "F:"),
              "last-sentence": option.replace("m :", "M:").replace("f :", "F:"),
              "label": 0
          }
          negative_corpus.append(obj)

# corpus[-1]
print(positive_corpus[:5])
print(negative_corpus[:5])
print(len(positive_corpus))
print(len(negative_corpus))

for example in dream_train:
  article = " ".join(example[0][:-1])
  last_sentence = example[0][-1].replace("W:", "F:")
  obj = {
      "id": f"dream-{example[2]}",
      "article": article,
      "label": 1,
      "QA": [x['question'] + ' ' + x['answer'] for x in example[1]],
      "last-sentence": last_sentence,
  }
positive_corpus.append(obj)

# corpus[-1]
# len(corpus)
positive_corpus[-5:]

for k, v in tc.items():
    lines = []
    for i, c in enumerate(v['content']):
        if i % 2 == 0:
            prefix = "M: "
        else:
            prefix = "F: "
        line = prefix + " ".join(c['message'])
        lines.append(line)
    obj = {
        "id": f"topical-chat-{k}",
        "article": " ".join(lines[:-1]),
        "last-sentence": lines[-1],
        "label": 1
    }
    positive_corpus.append(obj)

# len(tc)
positive_corpus[-5:]

"""## Permute utterance (reverse label)"""

def permute_utterance(article):
    article = re.split(r"(M: |F: )", article)
    article = [article[i] for i in range(0, len(article), 2) if len(article[i]) > 0]
    idx = np.arange(len(article))
    while np.array_equal(idx, np.arange(len(article))):
        idx = np.random.permutation(idx)
    new_article = []
    for i, x in enumerate(idx):
        if i % 2 == 0:
            prefix = "M: "
        else:
            prefix = "F: "
        new_article.append(prefix + article[x])
    new_article = " ".join(new_article)
    return new_article

permute_utterance("M: Hello! F: Hi!")

"""## Remove random utterance (reverse label)"""

def remove_utterance(article):
    article = re.split(r"(M: |F: )", article)
    article = [article[i] for i in range(0, len(article), 2) if len(article[i]) > 0]
    idx = np.arange(len(article))
    if len(article) <= 2:
        return None
    drop_num = np.random.randint(1, len(article) - 1)
    keep_idx = np.random.choice(idx[1: -1], size=len(article) - 2 - drop_num, replace=False)
    keep_idx = np.sort(keep_idx)
    idx = np.concatenate([idx[:1], keep_idx, idx[-1:]])
    new_article = []
    for i, x in enumerate(idx):
        if i % 2 == 0:
            prefix = "M: "
        else:
            prefix = "F: "
        new_article.append(prefix + article[x])
    new_article = " ".join(new_article)
    return new_article

from pprint import pprint
print(corpus[17000]['article'])
print(remove_utterance(corpus[17000]['article']))

"""## Replace random utterance (reverse label)"""

def replace_utterance(article, uid):
    article = re.split(r"(M: |F: )", article)
    article = [article[i] for i in range(0, len(article), 2) if len(article[i]) > 0]
    # TODO: get a random utterance based on tf-idf score
    # random number to replace
    rn = np.random.randint(1, len(article) // 2 + 1)
    # print(rn)
    ridx = np.random.choice(len(article), replace=False, size=rn)
    for i in ridx:
        p = id_map[f"{uid}-{i}"]
        sims_idx = cosine_similarity(X[p], X).argsort()[0][::-1][:10]
        cands = []
        for s in sims_idx:
            if uid in id_map_r[s]:
                continue
            cands.append(s)
        try:
          replace_ut = utterances[np.random.choice(cands)]
        except ValueError:
          return None
        # print(i, article[i], replace_ut)
        article[i] = replace_ut
        

    idx = np.arange(len(article))
    new_article = []
    for i, x in enumerate(idx):
        if i % 2 == 0:
            prefix = "M: "
        else:
            prefix = "F: "
        new_article.append(prefix + article[x])
    new_article = " ".join(new_article)
    return new_article

def replace_last_utterance(article, uid):
    article = re.split(r"(M: |F: )", article)
    article = [article[i] for i in range(0, len(article), 2) if len(article[i]) > 0]

    p = id_map[f"{uid}-{len(article)-1}"]
    sims_idx = cosine_similarity(X[p], X).argsort()[0][::-1][10:min(20, len(id_map))]
    cands = []
    for s in sims_idx:
        if uid in id_map_r[s]:
            continue
        cands.append(s)
    try:
      replace_ut = utterances[np.random.choice(cands)]
    except ValueError:
      return None
    article[len(article)-1] = replace_ut
        

    idx = np.arange(len(article))
    new_article = []
    for i, x in enumerate(idx):
        if i % 2 == 0:
            prefix = "M: "
        else:
            prefix = "F: "
        new_article.append(prefix + article[x])
    new_article_ = " ".join(new_article[:-1])
    last_sentence = new_article[-1]
    return new_article_, last_sentence

"""## Truncate utterance (keep label)"""

def truncate_utterance(article):
    article = re.split(r"(M: |F: )", article)
    article = [article[i] for i in range(0, len(article), 2) if len(article[i]) > 0]
    # randomly remove ending utterances (keep at least half of the utterances)
    if len(article) <= 2:
        return None
    new_article_len = np.random.randint(len(article)//2, len(article))
    start_idx = np.random.randint(0, len(article) - new_article_len)
    if start_idx % 2 == 1:
      start_idx -= 1
    end_idx = start_idx + new_article_len

    idx = np.arange(start_idx, end_idx)
    new_article = []
    for i, x in enumerate(idx):
        if i % 2 == 0:
            prefix = "M: "
        else:
            prefix = "F: "
        new_article.append(prefix + article[x])
    new_article = " ".join(new_article)
    return new_article

truncate_utterance("M: Hello! F: Hi! M: How are you doing? F: Fine! And you?")

tempp = []
for obj in positive_corpus:
  if "F:" in obj['article'] and "M:" in obj['article']:
    tempp.append(obj)
positive_corpus = tempp

tempn = []
for obj in negative_corpus:
  if "F:" in obj['article'] and "M:" in obj['article']:
    tempn.append(obj)
negative_corpus = tempn

print(len(positive_corpus))
print(len(negative_corpus))

"""Add truncate objects to positive corpus

"""

import random
truncated_positive_corpus = []
i = 0
while i < 4348:
  obj = random.choice(positive_corpus[-7000:])
  new_article = truncate_utterance(obj["article"])
  if new_article is None:
    continue
  new_article = re.split(r"(f : |m : |M: |F: )", new_article)
  new_obj = {
      "id": obj["id"] + "-trunc",
      "article": "".join(new_article[:-2]),
      "last-sentence": "".join(new_article[-2:]),
      "label": 1
  }
  truncated_positive_corpus.append(new_obj)
  i += 1


# print(len(truncated_positive_corpus))
# print(truncated_positive_corpus)
positive_corpus = positive_corpus + truncated_positive_corpus
print(len(positive_corpus))
# print(positive_corpus[0])

art = "F: bread , milk , eggs , i think that 's all we need . so do you want to come with me ? M: sure , i 'll drive . do n't forget to write down sugar and chocolate though . they are the most important materials . F: sugar and chocolate , right ? no problem . i have added them to my list and i 'll highlight them as the most important ."
import re
print("".join(re.split(r"(f : |m : |M: |F: )", art)[-2:]))

# prepare tf-idf matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

utterances = []
id_map = {}
id_map_r = {}
gid = 0
for c in positive_corpus:
    article = re.split(r"(M: |F: )", c["article"])
    article = [article[i] for i in range(0, len(article), 2) if len(article[i]) > 0]
    for i, a in enumerate(article):
        id_map[f"{c['id']}-{i}"] = gid
        id_map_r[gid] = f"{c['id']}-{i}"
        utterances.append(a)
        gid += 1

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(utterances)

# negative_corpus = []
# not_mutual_positive = [obj for obj in positive_corpus if "mutual" not in obj['id']]
# while len(negative_corpus) < 25000:
#   obj = random.choice(not_mutual_positive)
#   method = random.choice(["permute","remove","replace"])
#   if method == "permute":
#     article = permute_utterance(obj["article"])
#   elif method == "remove":
#     article = remove_utterance(obj["article"])
#   elif method == "replace":
#     article = replace_utterance(obj["article"], obj["id"])
#   if article is None:
#     continue
#   new_obj = {
#       "id": obj["id"],
#       "article": article,
#       "last-sentence": "".join(re.split(r"(f : |m : |M: |F: )", article)[-2:]),
#       "label": 0
#   }
#   negative_corpus.append(new_obj)

# print(len(negative_corpus))
# print(negative_corpus[0])

negative_corpus = []
not_mutual_positive = [obj for obj in positive_corpus if "mutual" not in obj['id']]
while len(negative_corpus) < 8928:
  obj = random.choice(not_mutual_positive)
  article = re.split(r"(f : |m : |M: |F: )", obj["article"])
  article = " ".join(["".join(i) for i in zip(article[1:-2:2], article[2:-1:2])])
  article, last_sentence = replace_last_utterance(article, obj["id"])
  if article is None:
    continue
  new_obj = {
      "id": obj["id"],
      "article": article,
      "last-sentence": last_sentence,
      "label": 0
  }
  negative_corpus.append(new_obj)

print(len(negative_corpus))
print(negative_corpus[0])

replaced_negative_corpus = negative_corpus

negative_corpus = negative_corpus + replaced_negative_corpus
print(len(negative_corpus))

dataset_next_uttr = positive_corpus + negative_corpus
print(len(dataset_next_uttr))

with open("dataset_next_uttr.json", "w") as f:
  json.dump(dataset_next_uttr, f)

[obj for obj in positive_corpus if obj['id'] == 'topical-chat-t_d1e9d0ea-e718-4094-9dd7-34e76b471729-trunc']

# Commented out IPython magic to ensure Python compatibility.
# %mv /content/dataset_next_uttr.json /content/drive/MyDrive/datasets