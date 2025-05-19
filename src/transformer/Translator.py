''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask
import numpy as np

class TranslatorContinue(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, max_seq_len,
        device,tokenizer,threshold,cvae):
        super(TranslatorContinue, self).__init__()
        # self.alpha = 0.7
        self.max_seq_len = max_seq_len
        # self.src_pad_idx = src_pad_idx
        # self.trg_bos_idx = trg_bos_idx
        # self.trg_eos_idx = trg_eos_idx
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.cvae = cvae
        self.model.eval()

    def _model_decode(self, trg_exp, enc_output, src_mask):
        # trg_mask = get_subsequent_mask(trg_seq)
        bsz = trg_exp.size()[0]
        seq_len = trg_exp.size()[1] 
        # pad_list = ['[UNK]']*seq_len
        pad_list = [' '.join(['[UNK]']*seq_len)]
        tgt_text = pad_list*bsz
        tgt_encodings = self.tokenizer(
            tgt_text, max_length=seq_len, padding='max_length', truncation=True, return_tensors="pt"
        )

        trg_seq = tgt_encodings['input_ids'].to(self.device)
        # if seq_len <3:
        #     trg_seq = trg_seq[:, :seq_len]
        trg_mask = get_subsequent_mask(trg_seq).repeat(bsz,1,1)
        exp_mask = torch.ones(bsz, seq_len, dtype=torch.bool).to(self.device)
        exp_mask[:,0] = False
        if self.cvae:
            dec_output, *_ = self.model.decoder(trg_seq, trg_mask, trg_exp, exp_mask, enc_output, src_mask,'infer')
        else:
            dec_output = self.model.decoder(trg_seq, trg_mask, trg_exp, exp_mask, enc_output, src_mask,'infer')
        dec_output = self.model.trg_exp_prj(dec_output)
        return dec_output
        # return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    def _get_all_state(self,src_seq,src_mask,tgt_seq,tgt_mask,tgt_exp_mask,tgt_exp):
        enc_output = self.model.encoder(input_ids=src_seq, attention_mask=src_mask).last_hidden_state
        dec_output, *_ = self.model.decoder(tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask, enc_output, src_mask)
        dec_output = self.model.trg_exp_prj(dec_output)
        return dec_output
        # tokenizer = BertTokenizer.from_pretrained(opt.pretrained_path)
        # tgt_texts_train = [' '.join(['[UNK]'] * i) for i in tgt_exps_steps_train]
        # tgt_texts_dev = [' '.join(['[UNK]'] * i) for i in tgt_exps_steps_dev]
        # training_data = get_dataloader(src_texts_train, tgt_texts_train,tgt_exps_train, tokenizer, opt.batch_size,opt.src_len,opt.trg_len,opt.trg_exp_dim,shuffle=False)

    def _get_init_state(self, src_seq, src_mask):
        # beam_size = self.beam_size
        if self.model.encoder_pretrained:
            enc_output = self.model.encoder(input_ids=src_seq, attention_mask=src_mask).last_hidden_state
        else:

            enc_output, *_ = self.model.encoder(src_seq, src_mask) #[1,11,512]
        # dec_output = self._model_decode(self.init_seq, enc_output, src_mask) #dec_output [1,50]
        # trg_mask = get_subsequent_mask(self.init_seq)
        # dec_output, *_ = self.model.decoder(self.init_seq, trg_mask, enc_output, src_mask)
        bsz = src_seq.size()[0]
        tgt_text = ['[UNK] [UNK]']*bsz
        tgt_encodings = self.tokenizer(
            tgt_text, max_length=3, padding='max_length', truncation=True, return_tensors="pt"
        )
        trg_seq = tgt_encodings['input_ids'][:,:2].to(self.device)
        trg_mask = get_subsequent_mask(trg_seq).repeat(bsz,1,1)
        trg_exp = torch.zeros(bsz,2,53).to(self.device) # will not use this value
        exp_mask = torch.zeros(bsz, 2, dtype=torch.bool).to(self.device) # set false, so this value will disabled
        exp_mask[:,1] = True
        mean = np.load('stage1_mean.npy')  #(53,)
        dec_output = torch.tensor(mean).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 53).to(self.device)


        padding = torch.zeros(bsz,1,53).to(self.device)
        dec_output = torch.cat((padding,dec_output,padding),dim=1) 
        # print(dec_output.size())
        return enc_output,dec_output

    def eos_in_all(self,gen_exp,no_end_index):
        bsz, seq, dim = gen_exp.shape
        zero_vector = torch.zeros(dim, device=gen_exp.device)
        close_indices = []
        if seq ==1:
            return
        for i in no_end_index:
            last_token = gen_exp[i, -1, :]
            distance = torch.norm(last_token - zero_vector)
            if distance < self.threshold:
                close_indices.append(i)
        for item in close_indices:
            no_end_index.remove(item)

    def serial_inference(self, src_seq,src_mask):
        '''
        '''
        max_seq_len = self.max_seq_len
        bsz = src_seq.size()[0]
        with torch.no_grad():
            enc_output,gen_seq = self._get_init_state(src_seq, src_mask)
            # enc_output, gen_seq = self._get_all_state(src_seq,src_mask,tgt_seq,tgt_mask,tgt_exp_mask,tgt_exp)
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq, enc_output, src_mask) #[bsz, step, 50]
                padding = torch.zeros(bsz,1,53).to(self.device)
                gen_seq = torch.cat((gen_seq[:,:-1,:],dec_output[:,-2,:].unsqueeze(1),padding),dim=1)

                # self.eos_in_all(gen_seq,no_end_index)
                # if no_end_index==[]:
                    # break
        return gen_seq[:,1:-1,:] 
    
    def parallel_inference(self,src_seq,src_mask,tgt_seq,tgt_mask,tgt_exp_mask,tgt_exp):
        with torch.no_grad():
            if self.cvae:
                pred,kl_loss,pred_pri_z = self.model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask,'infer')
            else:
                pred = self.model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask,'infer')
            return pred
