# PyTorch implementation of "Learning Compositional Rules via Neural Program Synthesis"

This is the codebase for the following paper:
_Learning Compositional Rules via Neural Program Synthsis_\
Maxwell I. Nye, Armando Solar-Lezama, Joshua B. Tenenbaum, Brenden M. Lake\
[https://arxiv.org/pdf/2003.05562.pdf](https://arxiv.org/pdf/2003.05562.pdf)

Much of this code is based on the [meta seq2seq](https://github.com/facebookresearch/meta_seq2seq) code by Brenden Lake.

## Requirements

Python 3.7

PyTorch 1.4.0

PHP intl (installed via `sudo apt install php7.0-intl` on ubuntu 16.04)

pyro (`pip3 install pyro-ppl`)

[pyprob](https://github.com/pyprob/pyprob) (installed from source)

Add necessary folders:
```
mkdir out_models
mkdir results
mkdir logs
mkdir testnums
```

We use `zsh`, though `bash` should also work for running the `.sh` scripts.

## MiniSCAN experiments

to train synthesis network:
```
python synthTrain.py --fn_out_model 'miniscan_final.p' --batchsize 128 --episode_type 'rules_gen'
```

to train meta seq2seq network:
```
python train_metanet_attn.py --fn_out_model 'metas2s_baseline.p' --episode_type 'rules_gen'
```

to run evaluation:
```
zsh miniscan_test.sh
```

to run evaluation of human tested domain in Figure 2:
```
zsh human_miniscan.sh
```


## SCAN experiments

to train synthesis network:
```
python synthTrain.py --fn_out_model 'scan_final.p' --batchsize 128 --episode_type 'scan_random' --num_pretrain_episodes 1000000
```

to train meta seq2seq network:
```
python train_metanet_attn.py --num_episodes 10000000 --fn_out_model 'scan_metas2s_baseline.p' --episode_type 'scan_random'
```

to replicate baselines in Table 1:
```
zsh SCAN_baselines.sh
```

to replicate results of full synthesis model with search (Table 1, top row) and search budget details (Table 2):
```
zsh scan_search_run.sh
```


to replicate results of full synthesis model with fixed example sets (Supplement Table 6):
```
zsh SCAN_fixed_budget.sh
```

## Number word experments

to train synthesis network:
```
python synthTrain.py --episode_type wordToNumber --type WordToNumber --print_freq 50 --batchsize 128 --save_freq 150 --fn_out_model WordToNum.p 
```

to train meta seq2seq network:
```
python train_metanet_attn.py --fn_out_model 'MetaNetw2num.p' --episode_type 'wordToNumber'
```

to run evaluation:
```
zsh number_test.sh
```
