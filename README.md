# VHM
Code for the ACL2022 main conference paper [A Variational Hierarchical Model for Neural Cross-Lingual Summarization](https://arxiv.org/abs/2203.03820).

## Introduction
In this work, we mainly bulid a variational hierarchical model via conditional variational auto-encoders that introduce a global variable
to combine the local ones for translation and summarization at the same time for CLS. The implementation is mainly based on [NCLS-Corpora](https://github.com/ZNLP/NCLS-Corpora). The dataset we used is from the [NCLS](https://aclanthology.org/D19-1302/) and then change the path in 'run_config/*.json' correspondingly.

## Requirements

+ pytorch >= 1.0
+ python 3.6

## Usage

Training with the following scripts: 

+ Build vocab

```
cd tools 
python -u build_vocab $number < $textfile > $vocab_file
```

+ Start training

```
python -u train.py -config run_config/train-zhen.json -batch_size $batch_size -kl_annealing_steps $kl_steps -latent_dim $latent_dim
```


+ Start decoding

```
python -u translate.py -config run_config/decode-zhen.json
```

## Citation

If you find this project helps, please cite our paper :)

```
@article{liang2022variational,
  title={A Variational Hierarchical Model for Neural Cross-Lingual Summarization},
  author={Liang, Yunlong and Meng, Fandong and Zhou, Chulun and Xu, Jinan and Chen, Yufeng and Su, Jinsong and Zhou, Jie},
  journal={arXiv preprint arXiv:2203.03820},
  year={2022}
}
```
Please feel free to open an issue or email me (yunlonliang@gmail.com) for questions and suggestions.
