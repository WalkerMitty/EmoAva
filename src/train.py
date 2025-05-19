'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import torch.optim as optim
# from torchtext.data import Field, Dataset, BucketIterator
# from torchtext.datasets import TranslationDataset

# import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from my_dataloader import load_dataset_split,get_dataloader,load_dataset_split_batch
from transformers import BertTokenizer,BertModel


def cal_loss_continue(pred, gold, gold_mask_o,return_first=False):
    '''
    :param pred: [bsz, seq-1, 50]
    :param gold: [bsz, seq-1, 50]
    :param gold_mask: [bsz, seq-1]
    :return: 
    '''
    weights = torch.ones(53).to(gold.device)
    weights[-3:] = 5.0  
    pred = pred * weights
    gold = gold * weights  
    euclidean_distance = torch.sqrt(torch.sum((pred - gold) ** 2, dim=-1))
    gold_mask = gold_mask_o.clone()
    gold_mask[:,0] = False
    masked_euclidean_distance = euclidean_distance[gold_mask] 
    
    if masked_euclidean_distance.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    loss = torch.mean(masked_euclidean_distance)
    if return_first:
        return loss,torch.mean(euclidean_distance[:,0])
    return loss

def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device):
    ''' Epoch operation in training phase'''

    model.train()
    # total_loss, n_word_total, n_word_correct = 0, 1, 0
    all_batchs = 0
    total_loss_kl = 0
    total_loss_dis = 0
    total_loss_one_step = 0
    total_first_token_loss = 0
    kl_loss = 0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):


        src_seq = batch['src_input_ids'].to(device)
        src_mask = batch['src_attention_mask'].to(device)
        tgt_seq = batch['tgt_input_ids'].to(device)
        tgt_mask = batch['tgt_attention_mask'].to(device)
        tgt_exp_mask = batch['tgt_exp_mask'].to(device)
        tgt_exp = batch['tgt_exp'].to(device)
        gold =batch['tgt_exp_gold'].to(device)
        gold_mask = batch['tgt_gold_mask'].to(device)

        optimizer.zero_grad()
        # pred = model(src_seq, trg_seq)
        if opt.cvae:
            pred,kl_loss,pred_pri_z = model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask,'train')
            loss,first_token_loss = cal_loss_continue(pred, gold, gold_mask,True)
            total_first_token_loss +=first_token_loss.item()
            total_loss = kl_loss +loss
            if opt.one_step_loss:
                one_step_loss = cal_loss_continue(pred_pri_z,gold[:,:-1,:],gold_mask[:,:-1])
                total_loss += one_step_loss
                total_loss_one_step +=one_step_loss.item()

            total_loss.backward()
            total_loss_kl+=kl_loss.item()
        else:
            pred = model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask,'train')
            loss = cal_loss_continue(pred, gold, gold_mask)
            total_loss = loss
            total_loss.backward()

        optimizer.step_and_update_lr()

        # note keeping
        # n_word_total += n_word
        # n_word_correct += n_correct
        total_loss_dis += loss.item()


            
        all_batchs+=1

    # loss_per_word = total_loss/n_word_total
    # accuracy = n_word_correct/n_word_total
    loss_per_batch = total_loss_dis/all_batchs
    # return loss_per_word, accuracy
    return loss_per_batch,total_loss_kl/all_batchs,total_loss_one_step/all_batchs, total_first_token_loss/all_batchs

def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss = 0
    total_first_token_loss = 0
    all_batchs = 0
    # total_loss_kl = 0
    # kl_loss = 0
    # total_one_step_loss = 0
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            # src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            # trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))
            src_seq = batch['src_input_ids'].to(device)
            src_mask = batch['src_attention_mask'].to(device)
            tgt_seq = batch['tgt_input_ids'].to(device)
            tgt_mask = batch['tgt_attention_mask'].to(device)
            tgt_exp_mask = batch['tgt_exp_mask'].to(device)
            tgt_exp = batch['tgt_exp'].to(device)
            gold = batch['tgt_exp_gold'].to(device)
            gold_mask = batch['tgt_gold_mask'].to(device)
            # forward
            # pred = model(src_seq, trg_seq)
            if opt.cvae:
                pred,kl_loss,pred_pri_z = model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask,'eval')
                # loss, n_correct, n_word = cal_performance(
                #     pred, gold, opt.trg_pad_idx, smoothing=False)
                loss,first_token_loss = cal_loss_continue(pred, gold, gold_mask,True)
                total_first_token_loss += first_token_loss.item()
            else:
                pred = model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask,'eval')
                loss = cal_loss_continue(pred, gold, gold_mask)

            total_loss += loss.item()
            

            all_batchs+=1
    # loss_per_word = total_loss/n_word_total
    # accuracy = n_word_correct/n_word_total
    # return loss_per_word, accuracy
    return total_loss/all_batchs,total_first_token_loss/all_batchs

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    # log_train_file = os.path.join(opt.output_dir, 'train.log')
    # log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    # print('[Info] Training performance will be written to file: {} and {}'.format(
    #     log_train_file, log_valid_file))

    # with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
    #     log_tf.write('epoch,loss,accuracy\n')
    #     log_vf.write('epoch,loss,accuracy\n')

    def print_performances_train(header, dis_loss, kl_loss, start_time, lr,one_step_loss,first_token_loss):
        print('  - {header:12} first_token_loss: {first_token_loss: 8.5f}, dis_loss: {dis_loss: 8.5f}, kl_loss: {kl_loss:8.5f} , onestep_loss: {one_step_loss:8.5f}, lr: {lr:8.6f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})",first_token_loss=first_token_loss, dis_loss=dis_loss,
                  kl_loss=kl_loss,one_step_loss=one_step_loss, elapse=(time.time()-start_time)/60, lr=lr))
        
    def print_performances_eval(header, dis_loss,start_time, lr,first_token_loss):
        print('  - {header:12} first_token_loss: {first_token_loss: 8.5f}, dis_loss: {dis_loss: 8.5f}, lr: {lr:8.6f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", first_token_loss=first_token_loss,dis_loss=dis_loss,
                elapse=(time.time()-start_time)/60, lr=lr))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        dis_loss, kl_loss, one_step_loss,first_token_loss = train_epoch(
            model, training_data, optimizer, opt, device)
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances_train('Training', dis_loss, kl_loss,start, lr,one_step_loss,first_token_loss)
        if (epoch_i+1) % opt.eval_interval==0:
            start = time.time()
            valid_dis_loss,valid_first_loss = eval_epoch(model, validation_data, device, opt)

            # valid_ppl = math.exp(min(valid_loss, 100))
            print_performances_eval('Validation', valid_dis_loss, start, lr,valid_first_loss)

            valid_losses += [valid_dis_loss]

            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

            if opt.save_mode == 'all':
                model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=valid_losses[-1])
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = 'model.chkpt'
                if valid_dis_loss <= min(valid_losses):
                    torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                    print('    - [Info] The checkpoint file has been updated.')
            
            if opt.use_tb:
                tb_writer.add_scalars('eval loss', {'dis_loss': valid_dis_loss}, epoch_i)
                tb_writer.add_scalars('eval loss', {'first_token_loss': valid_first_loss}, epoch_i)
            



        if opt.use_tb:
            tb_writer.add_scalars('train loss', {'dis_loss': dis_loss, 'kl_loss': kl_loss,'onestep_loss':one_step_loss, 'first_token_loss':first_token_loss}, epoch_i)
            # tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

def main():
    '''
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    # parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    # parser.add_argument('-train_path', default=None)   # bpe encoded data
    # parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=12)
    # parser.add_argument('-n_head_fea', type=int, default=8,help='Feature attention n_head_fea *n_k = trg_len')  
    parser.add_argument('-n_layers', type=int, default=1,help='control decoder layer')
    parser.add_argument('-n_layers_e', type=int, default=12,help='control encoder layer')   
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=42)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb', action='store_true', help='whether or not to scale the word / exp embedding at first, there is no meaning when pretrained encoder is used')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-cvae',action='store_true',help='Whether to use CVAE-version transformer')
    # parser.add_argument('-train_mode', type=str,choices=['train','eval','infer'],help='Only works when cvae is true')
    parser.add_argument('-flat_method', type=str,choices=['flat','casual_attention','none'],help='The different process of latent code z')
    parser.add_argument('-feature_attention', action='store_true',help='Whether to use Feature attention of Jaw code')
    parser.add_argument('-trg_exp_dim', type=int, default=53,help='exp dim, 50+3 in EMOCA, exp 50 , jaw 3')
    parser.add_argument('-exp_only_dim', type=int, default=50,help='exp only dim, 50 in EMOCA')
    parser.add_argument('-encoder_pretrained', type=str, choices=['bert-base','none'],help='use the different versions of encoder')
    parser.add_argument('-pretrained_path', type=str, default='/data/hfmodel/bert_base_case',help='the path of tokenizer and pretrained BERT series encoder')
    parser.add_argument('-one_step_loss', action='store_true',help='Whether or not to use one_step_loss, which can alleviate the problem of model collapse')
    parser.add_argument('-src_len', type=int, default=128)
    parser.add_argument('-trg_len', type=int, default=512)  
    parser.add_argument('-fix_encoder', action='store_true',help='fix encoder parameters')
    parser.add_argument('-eval_interval', type=int, default=200)
    parser.add_argument('-second_sample', type=str, choices=['one','none','learn'],help='the method of second sample',default='none')
    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    if opt.cvae:
        print('This is CVAE mode ...')
    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    # if opt.seed is not None:
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    np.random.seed(opt.seed)
    random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
    #     print('[Warning] The warmup steps may be not enough.\n'\
    #           '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
    #           'Using smaller batch w/o longer warmup may cause '\
    #           'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    # if all((opt.train_path, opt.val_path)):
    #     training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    # elif opt.data_pkl:
    src_texts_train, tgt_exps_train, tgt_exps_steps_train = load_dataset_split_batch('train')
    src_texts_dev, tgt_exps_dev, tgt_exps_steps_dev = load_dataset_split_batch('dev')
    tokenizer = BertTokenizer.from_pretrained(opt.pretrained_path)
    tgt_texts_train = [' '.join(['[UNK]'] * i) for i in tgt_exps_steps_train]
    tgt_texts_dev = [' '.join(['[UNK]'] * i) for i in tgt_exps_steps_dev]
    training_data = get_dataloader(src_texts_train, tgt_texts_train,tgt_exps_train, tokenizer, opt.batch_size,opt.src_len,opt.trg_len,opt.trg_exp_dim,shuffle=True)
    validation_data = get_dataloader(src_texts_dev, tgt_texts_dev, tgt_exps_dev,tokenizer,opt.batch_size, opt.src_len, opt.trg_len,opt.trg_exp_dim)
    opt.src_vocab_size = tokenizer.vocab_size

    opt.src_pad_idx = 0
    opt.trg_pad_idx = 0
    # else:
        # raise

    print(opt)
    if opt.encoder_pretrained=='bert-base':
        opt.encoder_pretrained = BertModel.from_pretrained(opt.pretrained_path)
    elif opt.encoder_pretrained=='none':
        opt.encoder_pretrained = None
    else:
        raise NotImplementedError()
    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_exp_dim,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_layers_e = opt.n_layers_e,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb=opt.scale_emb,
        cvae=opt.cvae,
        flat_method=opt.flat_method,
        feature_attention=opt.feature_attention,
        encoder_pretrained=opt.encoder_pretrained,
        one_step_loss=opt.one_step_loss,
        exp_only_dim = opt.exp_only_dim,
        src_len = opt.src_len,
        trg_len = opt.trg_len,
        fix_encoder = opt.fix_encoder,
        second_sample = opt.second_sample
        ).to(device)
    


    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)



if __name__ == '__main__':
    main()
