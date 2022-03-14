# -*- coding: utf-8 -*-

from beaver.data.dataset import TranslationDataset
from beaver.data.dataset_wrapper import Dataset
from beaver.data.field import Field
from beaver.data.dataset_sum import SumTransDataset

def build_dataset(opt, data_path, vocab_path, device, train=True):
    task1_source_path = data_path[0]
    task1_target_path = data_path[1] # mt
    task2_source_path = data_path[2]
    task2_target_path = data_path[3] # mls
    task3_source_path = data_path[4]
    task3_target_path = data_path[5] # cls

#    source_field = Field(unk=True, pad=True, bos=True, eos=True)
#    translation_target_field = Field(unk=True, pad=True, bos=True, eos=True)
#    summary_target_field = Field(unk=True, pad=True, bos=True, eos=True)
#    mono_target_field = Field(unk=True, pad=True, bos=True, eos=True)

    source_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
    translation_target_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
    summary_target_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)
    mono_target_field = Field(unk=True, pad=True, bos=True, eos=True, ms=True, mt=True, cls=True)

    source_vocab, target_vocab = vocab_path
    source_special = source_field.special
    translation_target_special = translation_target_field.special
    summary_target_special = summary_target_field.special
    mono_target_special = mono_target_field.special

    with open(source_vocab, encoding="UTF-8") as f:
        source_words = [line.strip() for line in f]
    with open(target_vocab, encoding="UTF-8") as f:
        target_words = [line.strip() for line in f]

    if opt.mono:
        # source和摘要target共享词表
        source_special = mono_target_special = sorted(set(source_special + mono_target_special))

    source_field.load_vocab(source_words, source_special)
    translation_target_field.load_vocab(target_words, translation_target_special)
    mono_target_field.load_vocab(source_words, mono_target_special)
    summary_target_field.load_vocab(target_words, summary_target_special)

    data1 = TranslationDataset(task1_source_path, task1_target_path, opt.batch_size, device, train,
                               {'src': source_field, 'tgt': summary_target_field}, "MT") # MT
#    data2 = SumTransDataset(task2_source_path, task2_target_path, task3_target_path, opt.batch_size, device, train,
#                           {'src': source_field, 'tgt': translation_target_field, 'mono_tgt': mono_target_field})
    

    data2 = TranslationDataset(task2_source_path, task2_target_path, opt.batch_size, device, train,
                               {'src': source_field, 'tgt': mono_target_field}, "MS")

    data3 = TranslationDataset(task3_source_path, task3_target_path, opt.batch_size, device, train,
                               {'src': source_field, 'tgt': translation_target_field}, "CLS")

    return Dataset(data1, data2, data3) # mt mls cls
