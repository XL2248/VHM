# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn

from beaver.model.embeddings import Embedding
from beaver.model.transformer import Decoder, Encoder


class Generator(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        lsm_score = self.lsm(score)
        return lsm_score


class NMTModel(nn.Module):

    def __init__(self, encoder: Encoder,
                 task1_decoder: Decoder,
                 task1_generator: Generator,
                 model_opt):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.task1_decoder = task1_decoder
        self.task1_generator = task1_generator

        latent_dim = int(model_opt.latent_dim)
        half_latent_dim = int(latent_dim)
        # for mt latent code/ prior
        self.mu_mt_prior     = nn.Linear(512, latent_dim)
#        self.logvar_mt_prior = nn.Linear(512, latent_dim)

        # for mt latent code/ post
        self.mu_mt_post     = nn.Linear(1024, latent_dim)
#        self.logvar_mt_post = nn.Linear(1024, latent_dim)
        self.output_mt = nn.Linear(512 + latent_dim, 512)

        # for cls latent code/ prior
        self.mu_cls_prior     = nn.Linear(512 + 2 * latent_dim, latent_dim)
 #       self.logvar_cls_prior = nn.Linear(512 + 2 * latent_dim, latent_dim)

        # for cls latent code/ post
        self.mu_cls_post     = nn.Linear(1024 + 2 * latent_dim, latent_dim)
#        self.logvar_cls_post = nn.Linear(1024 + 2 * latent_dim, latent_dim)
        self.output_cls = nn.Linear(512 + half_latent_dim, 512)

        # for mls latent code/ prior
        self.mu_mls_prior     = nn.Linear(512, latent_dim)
#       self.logvar_mls_prior = nn.Linear(512, latent_dim)

        # for mt latent code/ post
        self.mu_mls_post     = nn.Linear(1024, latent_dim)
#        self.logvar_mls_post = nn.Linear(1024, latent_dim)
        self.output_mls = nn.Linear(512 + half_latent_dim, 512)

    def sample_z(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def sample_z1(self, mu, logvar):
        epsilon = logvar.new_empty(logvar.size()).normal_()
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                                   - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                                   - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld

    def softplus(self, x):
        return torch.log(1.0 + torch.exp(x))

    def forward(self, source, ori_target, flag, is_training=False):
        target = ori_target[:, :-1]  # shift left
        source_pad = source.eq(self.encoder.embedding.word_padding_idx)
        enc_out = self.encoder(source, source_pad)
        x_rep = torch.mean(enc_out, dim=1, keepdim=False)
        kl_mt_loss, kl_mls_loss, kl_cls_loss = 0, 0, 0

        if flag == 1:  # task1
            target_pad = target.eq(self.task1_decoder.embedding.word_padding_idx)
            decoder_outputs, _ = self.task1_decoder(target, enc_out, source_pad, target_pad)
            mu_mt_prior, logvar_mt_prior = self.mu_mt_prior(x_rep), self.softplus(self.mu_mt_prior(x_rep))#, self.logvar_mt_prior(x_rep)
            if is_training:
                target_pad = ori_target.eq(self.task1_decoder.embedding.word_padding_idx)
                y_out = self.encoder(ori_target, target_pad)
                y_rep = torch.mean(y_out, dim=1, keepdim=False)
                xy_rep = torch.cat((x_rep, y_rep), -1)
                mu_mt_post, logvar_mt_post = self.mu_mt_post(xy_rep), self.softplus(self.mu_mt_post(xy_rep)) #self.logvar_mt_post(xy_rep)
                z = self.sample_z(mu_mt_post, logvar_mt_post)
                # kl_loss
                kld1 = self.gaussian_kld(mu_mt_post, logvar_mt_post, mu_mt_prior, logvar_mt_prior)
                kl_mt_loss = torch.mean(kld1, dim=-1, keepdim=False)
            else:
                z = self.sample_z(mu_mt_prior, logvar_mt_prior)
            latent_z = torch.unsqueeze(z, 1)
            latent_z = latent_z.repeat(1, decoder_outputs.size()[1], 1)
            decoder_outputs = torch.cat((decoder_outputs, latent_z), -1)
            decoder_outputs = self.output_mt(decoder_outputs)
#            code.interact(local=locals())
            return self.task1_generator(decoder_outputs), kl_mt_loss
        elif flag == 2:  # task2
            target_pad = target.eq(self.task1_decoder.embedding.word_padding_idx)
            decoder_outputs, _ = self.task1_decoder(target, enc_out, source_pad, target_pad)
            mu_mls_prior, logvar_mls_prior = self.mu_mls_prior(x_rep), self.softplus(self.mu_mls_prior(x_rep)) #self.logvar_mls_prior(x_rep)
            if is_training:
                target_pad = ori_target.eq(self.task1_decoder.embedding.word_padding_idx)
                y_out = self.encoder(ori_target, target_pad)
                y_rep = torch.mean(y_out, dim=1, keepdim=False)
                xy_rep = torch.cat((x_rep, y_rep), -1)
                mu_mls_post, logvar_mls_post = self.mu_mls_post(xy_rep), self.softplus(self.mu_mls_post(xy_rep)) #self.logvar_mls_post(xy_rep)
                z = self.sample_z(mu_mls_post, logvar_mls_post)
                # kl_loss
                kld3 = self.gaussian_kld(mu_mls_post, logvar_mls_post, mu_mls_prior, logvar_mls_prior)
                kl_mls_loss = torch.mean(kld3, dim=-1, keepdim=False)
            else:
                z = self.sample_z(mu_mls_prior, logvar_mls_prior)
            latent_z = torch.unsqueeze(z, 1)
            latent_z = latent_z.repeat(1, decoder_outputs.size()[1], 1)
            decoder_outputs = torch.cat((decoder_outputs, latent_z), -1)
            decoder_outputs = self.output_mls(decoder_outputs)

            return self.task1_generator(decoder_outputs), kl_mls_loss
        else:
            target_pad = target.eq(self.task1_decoder.embedding.word_padding_idx)
            decoder_outputs, _ = self.task1_decoder(target, enc_out, source_pad, target_pad)
            mu_mt_prior, logvar_mt_prior = self.mu_mt_prior(x_rep), self.softplus(self.mu_mt_prior(x_rep)) #self.logvar_mt_prior(x_rep)
            z_mt = self.sample_z(mu_mt_prior, logvar_mt_prior)
            mu_mls_prior, logvar_mls_prior = self.mu_mls_prior(x_rep), self.softplus(self.mu_mls_prior(x_rep))# self.logvar_mls_prior(x_rep)
            z_mls = self.sample_z(mu_mls_prior, logvar_mls_prior)
            xz_rep = torch.cat((x_rep, z_mt, z_mls), -1)
            mu_cls_prior, logvar_cls_prior = self.mu_cls_prior(xz_rep), self.softplus(self.mu_cls_prior(xz_rep)) #self.logvar_cls_prior(xz_rep)
            if is_training:
                target_pad = ori_target.eq(self.task1_decoder.embedding.word_padding_idx)
                y_out = self.encoder(ori_target, target_pad)
                y_rep = torch.mean(y_out, dim=1, keepdim=False)
                xy_rep = torch.cat((x_rep, y_rep), -1)
                mu_mt_post, logvar_mt_post = self.mu_mt_post(xy_rep), self.softplus(self.mu_mt_post(xy_rep)) #self.logvar_mt_post(xy_rep)
                z_mt = self.sample_z(mu_mt_post, logvar_mt_post)
                mu_mls_post, logvar_mls_post = self.mu_mls_post(xy_rep), self.softplus(self.mu_mls_post(xy_rep)) #self.logvar_mls_post(xy_rep)
                z_mls = self.sample_z(mu_mls_post, logvar_mls_post)
                # hier latent
                xyz_rep = torch.cat((xy_rep, z_mt, z_mls), -1)
                mu_cls_post, logvar_cls_post = self.mu_cls_post(xyz_rep), self.softplus(self.mu_cls_post(xyz_rep)) #self.logvar_cls_post(xyz_rep)
                z = self.sample_z(mu_cls_post, logvar_cls_post)
                # kl_loss
                kld2 = self.gaussian_kld(mu_cls_post, logvar_cls_post, mu_cls_prior, logvar_cls_prior)
                kl_cls_loss = torch.mean(kld2, dim=-1, keepdim=False)
            else:
                z = self.sample_z(mu_cls_prior, logvar_cls_prior)
            latent_z = torch.unsqueeze(z, 1)
            latent_z = latent_z.repeat(1, decoder_outputs.size()[1], 1)
            decoder_outputs = torch.cat((decoder_outputs, latent_z), -1)
            decoder_outputs = self.output_cls(decoder_outputs)
            return self.task1_generator(decoder_outputs), kl_cls_loss

    @classmethod
    def load_model(cls, model_opt,
                   pad_ids: Dict[str, int],
                   vocab_sizes: Dict[str, int],
                   checkpoint=None):
        source_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                     dropout=model_opt.dropout,
                                     padding_idx=pad_ids["src"],
                                     vocab_size=vocab_sizes["src"])
        # MT
        if model_opt.share_source_target_embedding:
            target_embedding_task1 = source_embedding
        else:
            target_embedding_task1 = Embedding(embedding_dim=model_opt.hidden_size,
                                               dropout=model_opt.dropout,
                                               padding_idx=pad_ids["task1_tgt"],
                                               vocab_size=vocab_sizes["task1_tgt"])
        '''
        if model_opt.share_mt_cls_embedding:
            target_embedding_task3 = target_embedding_task1
        else:
            target_embedding_task3 = Embedding(embedding_dim=model_opt.hidden_size,
                                              dropout=model_opt.dropout,
                                              padding_idx=pad_ids["task3_tgt"],
                                              vocab_size=vocab_sizes["task3_tgt"])
        if model_opt.mono:
            # 单语摘要，task1 share source embedding
            target_embedding_task2 = source_embedding
        else:
            target_embedding_task2 = Embedding(embedding_dim=model_opt.hidden_size,
                                              dropout=model_opt.dropout,
                                              padding_idx=pad_ids["task2_tgt"],
                                              vocab_size=vocab_sizes["task2_tgt"])
        '''
        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          source_embedding)

        task1_decoder = Decoder(model_opt.layers,
                                model_opt.heads,
                                model_opt.hidden_size,
                                model_opt.dropout,
                                model_opt.ff_size,
                                target_embedding_task1)

        task1_generator = Generator(model_opt.hidden_size, vocab_sizes["task1_tgt"])
#        task2_generator = Generator(model_opt.hidden_size, vocab_sizes["task2_tgt"])
#        task3_generator = Generator(model_opt.hidden_size, vocab_sizes["task3_tgt"])

#        model = cls(encoder, task1_decoder, task2_decoder, task3_decoder, task1_generator, task2_generator, task3_generator, model_opt)
        model = cls(encoder, task1_decoder, task1_generator, model_opt)
        if checkpoint is None and model_opt.train_from:
            checkpoint = torch.load(model_opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
