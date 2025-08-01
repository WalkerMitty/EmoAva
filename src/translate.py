import torch
import argparse
from tqdm import tqdm
import os
from transformer.Models import Transformer
from transformer.Translator import TranslatorContinue
from em_dataloader import load_dataset_split_batch,get_dataloader
from transformers import BertTokenizer
import pickle
import numpy as np 
import random
def load_model(in_opt, device):

    checkpoint = torch.load(in_opt.model, map_location=device)
    opt = checkpoint['settings']
    model = Transformer(
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
        second_sample=opt.second_sample
        ).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


def main():

    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', default='output/model.chkpt',
                        help='Path to model weight file')
    parser.add_argument('-tokenizer_path', type=str, default='/data/hfmodel/bert_base_case')
    parser.add_argument('-data_source', type=str, default='/data/hfmodel/bert_base_case')
    parser.add_argument('-save_path', type=str, default='infer_mid_results')
    parser.add_argument('-save_name', type=str, default='result.pt')
    parser.add_argument('-max_seq_len', type=int, default=256)
    parser.add_argument('-src_len', type=int, default=128)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-infer_mode', type=str,default='s',choices=['s','p'])
    parser.add_argument('-seed', type=int,default=42)
    parser.add_argument('-cvae',action='store_true')
    parser.add_argument('-uncondition', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    if os.path.exists(opt.save_path)==False:
        os.makedirs(opt.save_path)



    src_texts, tgt_exps, tgt_exps_steps = load_dataset_split_batch('test',5,None)
    tokenizer = BertTokenizer.from_pretrained(opt.tokenizer_path)
    tgt_texts = [' '.join(['[UNK]'] * i) for i in tgt_exps_steps]
   
    config = {
        'tokenizer':tokenizer,
        'batch_size':opt.batch_size,
        'src_len':opt.src_len,
        'trg_len':opt.max_seq_len,
        'shuffle':False,
    }
    test_loader = get_dataloader(src_texts, tgt_texts, tgt_exps,config)

    SEED = opt.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = TranslatorContinue(
        model=load_model(opt, device),
        max_seq_len=opt.max_seq_len,
        device=device,
        tokenizer=tokenizer,
        threshold=1.0,
        cvae=opt.cvae,
        data_source=opt.data_source,
        ).to(device)


    result_list = []
    for batch in tqdm(test_loader):

        src_seq = batch['src_input_ids'].to(device)
        src_mask = batch['src_attention_mask'].to(device)
        if opt.uncondition:
            src_seq.fill_(0)
            src_mask.fill_(False)
            print('Enter unconditional infer...')
        tgt_seq = batch['tgt_input_ids'].to(device)
        tgt_mask = batch['tgt_attention_mask'].to(device)
        tgt_exp_mask = batch['tgt_exp_mask'].to(device)
        tgt_exp = batch['tgt_exp'].to(device)

        if opt.infer_mode=='p':

            pred_exps = translator.parallel_inference(src_seq, src_mask,tgt_seq,tgt_mask,tgt_exp_mask,tgt_exp)
        elif opt.infer_mode=='s':
            pred_exps = translator.serial_inference(src_seq, src_mask)
        else:
            raise NotImplementedError()
        result_list.append(pred_exps)

    concat_tensors = torch.cat(result_list,dim=0)
    torch.save(concat_tensors,os.path.join(opt.save_path,opt.save_name))
    
def truncate_mask(save_path,threshold,filename):

    raw_predict  = torch.load(save_path).to('cpu')
    mask_tensor = torch.zeros(53)
    final_list = []
    for i in tqdm(range(raw_predict.size()[0])):
        find_mask = False
        for j in range(1,raw_predict.size()[1]):

            distance = torch.sqrt(torch.sum((mask_tensor - raw_predict[i][j]) ** 2, dim=-1))
            if distance<threshold:
                final_list.append(raw_predict[i,:j,:])
                find_mask = True
                break
        if not find_mask:
            final_list.append(raw_predict[i,:,:])
        
    with open(filename, 'wb') as f:
        pickle.dump(final_list, f)

if __name__ == "__main__":

    main()
