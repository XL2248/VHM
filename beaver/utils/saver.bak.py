import json

import torch
import os
import datetime


class Saver(object):
    def __init__(self, opt):
        self.opt = opt
        self.ckpt_names = []
        self.rouge1_results = []
        self.rouge2_results = []
#        self.model_path = opt.model_path + datetime.datetime.now().strftime("-%y%m%d-%H%M%S")
        self.model_path = opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split)
        self.max_to_keep = opt.max_to_keep
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        with open(os.path.join(self.model_path, "params.json"), "w", encoding="UTF-8") as log:
            log.write(json.dumps(vars(opt), indent=4) + "\n")

    def save(self, save_dict, step, loss_task1, loss_task2, loss_task3, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3, typ):

        with open(os.path.join(self.model_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t" % datetime.datetime.now())
            log.write("type: %s\t" % typ)
            log.write("step: %6d\t" % step)
            log.write("loss-mt: %.2f\t" % loss_task1)
            log.write("loss-mls: %.2f\t" % loss_task2)
            log.write("loss-cls: %.2f\t" % loss_task3)
            log.write("bleu-mt: %3.2f\t" % bleu_task1)
            log.write("bleu-mls: %3.2f\t" % bleu_task2)
            log.write("bleu-cls: %3.2f\t" % bleu_task3)
            log.write("rouge1-mt: %3.2f\t" % rouge1_task1)
            log.write("rouge1-mls: %3.2f\t" % rouge1_task2)
            log.write("rouge1-cls: %3.2f\t" % rouge1_task3)
            log.write("rouge2-mt: %3.2f\t" % rouge2_task1)
            log.write("rouge2-mls: %3.2f\t" % rouge2_task2)
            log.write("rouge2-cls: %3.2f\t" % rouge2_task3)
            log.write("\n")

        if typ == "valid":
            filename = "checkpoint-step-%06d" % step
            if self.opt.train_from and os.path.exists(os.path.join(self.model_path, "record")) and len(self.ckpt_names) == 0:
                with open(os.path.join(self.model_path, "record"), "r", encoding="UTF-8") as record:
                    content = record.readlines()
                    for line in content:
                        self.ckpt_names.append(line.strip().split("\t")[0].strip().split(": ")[1])
                        self.rouge1_results.append(float(line.strip().split("\t")[1].strip().split(": ")[1]))
                        self.rouge2_results.append(float(line.strip().split("\t")[2].strip().split(": ")[1]))

            full_filename = os.path.join(self.model_path, filename)
            self.ckpt_names.append(full_filename)
            self.rouge1_results.append(rouge1_task3)
            self.rouge2_results.append(rouge2_task3)
            torch.save(save_dict, full_filename)
            if len(self.ckpt_names) > self.max_to_keep:
                min_index = self.rouge2_results.index(min(self.rouge2_results))
                min_filename = self.ckpt_names[min_index]
                if os.path.exists(min_filename):
                    os.remove(min_filename)
                del self.rouge1_results[min_index]
                del self.rouge2_results[min_index]
                del self.ckpt_names[min_index]
                with open(os.path.join(self.model_path, "record"), "w", encoding="UTF-8") as record:
                    for idx in range(len(self.rouge2_results)):
                        record.write("checkpoint: %s\trouge1-cls: %3.2f\trouge2-cls: %3.2f\t" % (self.ckpt_names[idx], self.rouge1_results[idx], self.rouge2_results[idx]))
                        record.write("\n")
#            if 0 < self.max_to_keep < len(self.ckpt_names):
#                earliest_ckpt = self.ckpt_names.pop(0)
#                os.remove(earliest_ckpt)
