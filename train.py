# -*- coding: utf-8 -*-
import logging

import torch
import torch.cuda
import os
from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, printing_opt
from beaver.utils.metric import calculate_rouge

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Trainable', str(trainable_num), str(total_num))
    return {'Total': total_num, 'Trainable': trainable_num}

def valid(model, criterion_task1, criterion_task2, criterion_task3, valid_dataset, step, typ):
    model.eval()
    total_n = 0
    total_task1_loss = total_task2_loss =  total_task3_loss = 0.0
    task1_hypothesis, task1_references = [], []
    task2_hypothesis, task2_references = [], []
    task3_hypothesis, task3_references = [], []

    for i, (batch, flag) in enumerate(valid_dataset):

        scores, kl_loss = model(batch.src, batch.tgt, flag)
        _, predictions = scores.topk(k=1, dim=-1)
        if flag == 1:
            loss = criterion_task1(scores, batch.tgt)
        elif flag == 2:
            loss = criterion_task2(scores, batch.tgt)
        else: #if flag == 3:
            loss = criterion_task3(scores, batch.tgt)
#       _, predictions = scores.topk(k=1, dim=-1)

        if flag == 1:  # task1
            total_task1_loss += loss.data
            task1_hypothesis += [valid_dataset.fields["task1_tgt"].decode(p) for p in predictions]
            task1_references += [valid_dataset.fields["task1_tgt"].decode(t) for t in batch.tgt]
        elif flag == 2:
            total_task2_loss += loss.data
            task2_hypothesis += [valid_dataset.fields["task2_tgt"].decode(p) for p in predictions]
            task2_references += [valid_dataset.fields["task2_tgt"].decode(t) for t in batch.tgt]
        else:
            total_task3_loss += loss.data
            task3_hypothesis += [valid_dataset.fields["task3_tgt"].decode(p) for p in predictions]
            task3_references += [valid_dataset.fields["task3_tgt"].decode(t) for t in batch.tgt]

        total_n += 1

    bleu_task1 = calculate_bleu(task1_hypothesis, task1_references)
    bleu_task2 = calculate_bleu(task2_hypothesis, task2_references)
    bleu_task3 = calculate_bleu(task3_hypothesis, task3_references)
    rouge1_task1, rouge2_task1 = calculate_rouge(task1_hypothesis, task1_references)
    rouge1_task2, rouge2_task2 = calculate_rouge(task2_hypothesis, task2_references)
    rouge1_task3, rouge2_task3 = calculate_rouge(task3_hypothesis, task3_references)
    mean_task1_loss = total_task1_loss / total_n
    mean_task2_loss = total_task2_loss / total_n
    mean_task3_loss = total_task3_loss / total_n
    if typ == "test":
        with open(os.path.join(opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split), "prediction."+str(step)), "w", encoding="UTF-8") as out_file:
            out_file.write("\n".join(task3_hypothesis))
            out_file.write("\n")

    logging.info("type: %s \t loss-mt: %.2f \t loss-mls %.2f \t loss-cls %.2f \t bleu-mt: %3.2f\t bleu-mls: %3.2f \t bleu-cls: %3.2f \t rouge1-mt: %3.2f \t rouge1-mls: %3.2f \t rouge1-cls: %3.2f \t rouge2-mt: %3.2f \t rouge2-mls: %3.2f \t rouge2-cls: %3.2f"
                 % (typ, mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3, typ)


def train(model, criterion_task1, criterion_task2, criterion_task3, optimizer, train_dataset, valid_dataset, test_dataset):
    total_task1_loss = total_task2_loss = total_task3_loss = kl_mt_losses = kl_cls_losses = kl_mls_losses = 0.0
    model.zero_grad()

    for i, (batch, flag) in enumerate(train_dataset):
        kl_weights = min(optimizer.n_step * 1.0 / opt.kl_annealing_steps, 1.0)
        scores, kl_loss = model(batch.src, batch.tgt, flag, True)
#        logging.info(flag, batch.src.size(), batch.tgt.size(), scores.size())
        if flag == 1:
            loss = criterion_task1(scores, batch.tgt)
            kl_mt_loss = kl_loss
        elif flag == 2:
            loss = criterion_task2(scores, batch.tgt)
            kl_mls_loss = kl_loss
        else: # flag == 3:
            loss = criterion_task3(scores, batch.tgt)
            kl_cls_loss = kl_loss
        loss += kl_weights * kl_loss
        loss.backward()

        if flag == 1:  # task1
            total_task1_loss += loss.data
            kl_mt_losses += kl_mt_loss.data
        elif flag == 2:
            total_task2_loss += loss.data
            kl_mls_losses += kl_mls_loss.data
        else:
            total_task3_loss += loss.data
            kl_cls_losses += kl_cls_loss.data

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_task1_loss = total_task1_loss / opt.report_every / opt.grad_accum * 2
                mean_task2_loss = total_task2_loss / opt.report_every / opt.grad_accum * 2
                mean_task3_loss = total_task3_loss / opt.report_every / opt.grad_accum * 2
                mean_kl_mt = kl_mt_losses / opt.report_every / opt.grad_accum * 2
                mean_kl_cls = kl_cls_losses / opt.report_every / opt.grad_accum * 2
                mean_kl_mls = kl_mls_losses / opt.report_every / opt.grad_accum * 2
                logging.info("step: %7d\t loss-mt: %.4f \t loss-mls: %.4f \t loss-cls: %.4f \t kl_weights: %.8f \t kl-mt: %.4f \t kl-mls: %.4f \t kl-cls: %.4f"
                             % (optimizer.n_step, mean_task1_loss, mean_task2_loss, mean_task3_loss, kl_weights, mean_kl_mt, mean_kl_mls, mean_kl_cls))
                total_task1_loss = total_task2_loss = total_task3_loss = kl_mt_losses = kl_cls_losses = kl_mls_losses = 0.0

            if optimizer.n_step % opt.save_every == 0: # and optimizer.n_step > 100000:
                with torch.set_grad_enabled(False):
                    valid(model, criterion_task1, criterion_task2, criterion_task3, valid_dataset, optimizer.n_step, "valid")
                    if optimizer.n_step > 400000 and optimizer.n_step % (2 * opt.save_every) == 0:
                        valid(model, criterion_task1, criterion_task2, criterion_task3, test_dataset, optimizer.n_step, "test")
                model.train()
        if optimizer.n_step % 850000 == 0:
            logging.info("Training DONE all steps  %7d" % optimizer.n_step)
            break;
        del loss


def main():
    logging.info("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    test_dataset = build_dataset(opt, opt.test, opt.vocab, device, train=False)
    fields = valid_dataset.fields = train_dataset.fields = test_dataset.fields
    logging.info("Build model...")

    pad_ids = {"src": fields["src"].pad_id,
               "task1_tgt": fields["task1_tgt"].pad_id,
               "task2_tgt": fields["task2_tgt"].pad_id,
               "task3_tgt": fields["task3_tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab),
                   "task1_tgt": len(fields["task1_tgt"].vocab),
                   "task2_tgt": len(fields["task1_tgt"].vocab),
                   "task3_tgt": len(fields["task1_tgt"].vocab)}

#    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    criterion_task1 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task1_tgt"], pad_ids["task1_tgt"]).to(device)
    criterion_task2 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task2_tgt"], pad_ids["task2_tgt"]).to(device)
    criterion_task3 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task3_tgt"], pad_ids["task3_tgt"]).to(device)
    checkpoint_num = []
    if os.path.exists(opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split)):
        files= os.listdir(opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split))
        for fil in files:
            if not os.path.isdir(fil) and len(fil) > 20:
                checkpoint_num.append(int(fil.split("-")[-1]))
        if len(checkpoint_num) > 0:
            opt.train_from = opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split) + "/checkpoint-step-%06d" % max(checkpoint_num)
    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    logging.info("model parameters:", get_parameter_number(model))
    for name, parameters in model.named_parameters():
        print(name, ':', str(parameters.size()))
    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("start training...")
    train(model, criterion_task1, criterion_task2, criterion_task3, optimizer, train_dataset, valid_dataset, test_dataset)


if __name__ == '__main__':
    main()
