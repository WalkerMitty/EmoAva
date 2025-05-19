''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.SubLayers import MultiHeadAttention


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=128, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        #enc_output [256,29,512]
        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb: 
            enc_output *= self.d_model ** 0.5

        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        #[256,8,29,29] enc_slf_attn
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,n_trg_vocab, n_trg_dim, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=512, dropout=0.1, scale_emb=False,cvae=False,
            flat_method='casual_attention',feature_attention=False,one_step_loss=False,exp_only_dim=50,second_sample='none'):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.exp_emb = nn.Linear(n_trg_dim,d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.fea_attn = MultiHeadAttention(n_head, d_model, d_k, d_v) 
        self.fea_proj = nn.Linear(1,d_model)
        self.fea_proj_reverse = nn.Linear(d_model,1)
        self.dropout = nn.Dropout(p=dropout)
        self.exp_only_dim = exp_only_dim
        self.feature_attention = feature_attention
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,cvae=cvae,
                         flat_method=flat_method,one_step_loss=one_step_loss,second_sample=second_sample)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.n_layers = n_layers
        self.cvae = cvae

    # def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
    def forward(self, trg_seq, trg_mask, trg_exp, exp_mask, enc_output, src_mask,train_mode):
        '''
        :param trg_seq:
        :param trg_mask:  #[bsz,19,19]
        :param enc_output:
        :param src_mask: #[bsz, 1, 20]
        trg_exp : tensor [bsz, seq, 50]
        exp_mask: tensor [bsz, seq] 
        :param return_attns:
        :return:
        '''


        if self.cvae:
            if self.feature_attention:
                exp = torch.transpose(trg_exp[:,:,:self.exp_only_dim],1,2)  #[bsz, 50, seq_len]
                jaw = torch.transpose(trg_exp[:,:,self.exp_only_dim:],1,2)  #[bsz, 3, seq_len]
                temp_exp = self.fea_proj(torch.mean(exp,dim=2,keepdim=True))  #[bsz, 50, d_model]
                new_jaw = self.fea_proj(torch.mean(jaw,dim=2,keepdim=True))
                new_jaw,_ = self.fea_attn(new_jaw,temp_exp,temp_exp,mask=None)
                new_jaw = self.fea_proj_reverse(new_jaw)  #[bsz, 3, 1]
                trg_exp = torch.transpose(torch.concat((exp,new_jaw+jaw),dim=1),1,2)
            dec_output = self.trg_word_emb(trg_seq) #[bsz, seq, hidden]
            dec_output_exp = self.exp_emb(trg_exp) #[bsz, seq, hidden]
            dec_output[exp_mask] = dec_output_exp[exp_mask]

            if self.scale_emb:
                dec_output *= self.d_model ** 0.5
            dec_output = self.position_enc(dec_output)
            dec_output = self.layer_norm(dec_output)

            priz_predict = None
            total_layer_kl_loss = 0.0
            for id,dec_layer in enumerate(self.layer_stack):
                
                # last_priz_predict = priz_predict
                dec_output, kl_loss,priz_predict = dec_layer(
                    dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask,last_priz_predict=priz_predict,train_mode=train_mode)
                # dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                # dec_enc_attn_list += [dec_enc_attn] if return_attns else []
                if train_mode=='train':
                    total_layer_kl_loss= total_layer_kl_loss+kl_loss
            return dec_output,total_layer_kl_loss/self.n_layers,priz_predict
        
        
        else:
            if self.feature_attention:
                exp = torch.transpose(trg_exp[:,:,:self.exp_only_dim],1,2)  #[bsz, 50, seq_len]
                jaw = torch.transpose(trg_exp[:,:,self.exp_only_dim:],1,2)  #[bsz, 3, seq_len]
                temp_exp = self.fea_proj(torch.mean(exp,dim=2,keepdim=True))  #[bsz, 50, d_model]
                new_jaw = self.fea_proj(torch.mean(jaw,dim=2,keepdim=True))
                new_jaw,_ = self.fea_attn(new_jaw,temp_exp,temp_exp,mask=None)
                new_jaw = self.fea_proj_reverse(new_jaw)  #[bsz, 3, 1]
                trg_exp = torch.transpose(torch.concat((exp,new_jaw+jaw),dim=1),1,2)
            dec_output = self.trg_word_emb(trg_seq) #[bsz, seq, hidden]
            dec_output_exp = self.exp_emb(trg_exp) #[bsz, seq, hidden]
            dec_output[exp_mask] = dec_output_exp[exp_mask]

            if self.scale_emb:
                dec_output *= self.d_model ** 0.5
            dec_output = self.position_enc(dec_output)
            dec_output = self.layer_norm(dec_output)

            priz_predict = None
            # total_layer_kl_loss = 0.0
            for id,dec_layer in enumerate(self.layer_stack):
                
                # last_priz_predict = priz_predict
                dec_output = dec_layer(
                    dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask,last_priz_predict=priz_predict,train_mode=train_mode)

            return dec_output




class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_dim, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_layers_e=12,n_head=8, d_k=64, d_v=64, dropout=0.1, 
            trg_emb_prj_weight_sharing=True, 
            scale_emb=False,cvae=False,flat_method='casual_attention',
            feature_attention=False,encoder_pretrained=None,one_step_loss = False,
            exp_only_dim=50,src_len=128,trg_len=512,fix_encoder=False,second_sample='none'):
        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.d_model = d_model
        self.cvae = cvae
        self.encoder_pretrained = encoder_pretrained
        self.one_step_loss = one_step_loss
        if encoder_pretrained:
            self.encoder = encoder_pretrained
        else:
            self.encoder = Encoder(
                n_src_vocab=n_src_vocab, n_position=src_len,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers_e, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_src_vocab,
            n_trg_dim=n_trg_dim, n_position=trg_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb,
            cvae=cvae,flat_method=flat_method,
            feature_attention=feature_attention,one_step_loss=one_step_loss,exp_only_dim=exp_only_dim,second_sample=second_sample
            )

        # self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        self.trg_exp_prj = nn.Linear(d_model,n_trg_dim,bias=False)
        if one_step_loss:
            self.trg_word_prj_one_step = nn.Linear(d_model,n_trg_dim,bias=False)
        if encoder_pretrained is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            for n,p in self.named_parameters():
                # print(n)
                if 'encoder_pretrained' in n:
                    continue
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if fix_encoder:
            for n,p in self.encoder.named_parameters():
                p.requires_grad = False
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            # self.trg_word_prj.weight = self.decoder.exp_emb.weight.transpose(0,1)
            self.trg_exp_prj.weight = nn.Parameter(self.decoder.exp_emb.weight.transpose(0, 1).clone())


    def forward(self, src_seq, src_mask,trg_seq,trg_mask,trg_exp,exp_mask,train_mode):
        '''
        src_seq: bsz, src_len
        src_mask: bsz, 1, src_len
        trg_seq: bsz, trg_len-1     
        trg_mask: bsz, trg_len-1, trg_len-1,   
        trg_exp: bsz, trg_len-1, 53
        exp_mask: bsz, trg_len-1
        '''


        if self.encoder_pretrained is not None:
            enc_output = self.encoder(input_ids=src_seq, attention_mask=src_mask).last_hidden_state

        else:
            enc_output, *_ = self.encoder(src_seq, src_mask)

        if self.cvae:

            dec_output,kl_loss, priz_predict = self.decoder(trg_seq, trg_mask, trg_exp, exp_mask, enc_output,src_mask,train_mode) 

            seq_logit = self.trg_exp_prj(dec_output)
            if self.one_step_loss and train_mode!='infer':
                priz_predict_logit = self.trg_word_prj_one_step(priz_predict)
            else:
                priz_predict_logit = None
    
            
            return seq_logit,kl_loss,priz_predict_logit
        
        else:
            dec_output = self.decoder(trg_seq, trg_mask, trg_exp, exp_mask, enc_output,src_mask,train_mode)
            seq_logit = self.trg_exp_prj(dec_output)
            return seq_logit
