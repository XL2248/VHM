# -*- coding: utf-8 -*-
import itertools

from beaver.data.dataset import TranslationDataset
#from beaver.data.dataset_sum import SumTransDataset

class Dataset(object):
    def __init__(self, task1_dataset: TranslationDataset, task2_dataset: TranslationDataset, task3_dataset: TranslationDataset):
        self.task1_dataset = task1_dataset
        self.task2_dataset = task2_dataset
        self.task3_dataset = task3_dataset

        self.fields = {
            "src": task3_dataset.fields["src"],
            "task1_tgt": task1_dataset.fields["tgt"],
            "task2_tgt": task2_dataset.fields["tgt"],
            "task3_tgt": task3_dataset.fields["tgt"]
        }

    def __iter__(self):
        for batch1, batch2, batch3 in itertools.zip_longest(self.task1_dataset, self.task2_dataset, self.task3_dataset):
            if batch1 is not None:
                yield batch1, 1
            if batch3 is not None:
                yield batch3, 3
            if batch2 is not None:
                yield batch2, 2

