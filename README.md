# 11747-project

This repo is migrated from [MDFN](https://github.com/comprehensiveMap/MDFN)

## Requirements

```
transformers
tqdm

// To run MLM (masking), you need:
enchant
nltk
```

## Instructions

If you want to generate the dataset, follow `dataset.py`.

If you want to run the pre-training tasks, run:
```
sh scripts/mlm.sh
sh scripts/us.sh
sh scripts/nup.sh
```

After get the pre-trained model, use `decomp_weights` to get the weights saved in the `weights` folder. Change the weights loading path according to your specifications in the shell scripts file and run the experiments:

```
sh scripts/mutual_[mlm/us/nup].sh
```

If you want to run the option-comparison enhancement tasks, run:
```
python run_MDFN.py \
--data_dir datasets/mutual \
--model_name_or_path \
google/electra-large-discriminator \
--model_type [electra_v1/electra_v2/electra_v3] \
--task_name mutual\
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 6 \
--eval_batch_size 6 \
--learning_rate 4e-6 \
--num_train_epochs 3 \
--gradient_accumulation_steps 1 \
--local_rank -1 \
```


## Experiments

Download the data from https://github.com/Nealcly/MuTual.git and put them in data folder.

To reproduce MDFN, run:

```bash
chmod a+x scripts/train.sh
sh scripts/train.sh
```

To reproduce MDFN+, run:

```bash
chmod a+x scripts/train_plus.sh
sh scripts/train_plus.sh
```

## Analysis

Refer to `analysis/analysis.ipynb`.