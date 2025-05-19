''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward




class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,cvae=False,flat_method='casual_attention',
                 one_step_loss=False,second_sample='none'):
        super(DecoderLayer, self).__init__()


        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.enc_attn_second = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.second_sample = second_sample
        if self.second_sample=='learn':
            self.pos_ffn_var = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pri_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.cvae = cvae
        self.flat_method = flat_method
        self.one_step_loss = one_step_loss
        if self.cvae:
            self.upper_mu = nn.Linear(d_model,d_model)
            self.down_mu = nn.Linear(d_model,d_model)
            self.upper_sigma = nn.Linear(d_model,d_model)
            self.down_sigma = nn.Linear(d_model,d_model)
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.slf_attn_second = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            

    def sampling(self,z_mean, z_log_var):
        # epsilon is sampled from a normal distribution
        # import pdb;pdb.set_trace()
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(z_log_var / 2) * epsilon
    def obtain_kl(self,upper_mu,down_mu,upper_sigma,down_sigma):
        bsz,seq,hidden = upper_mu.shape
        # return kl_loss/num
        mu2 = down_mu[:,:-1,:].contiguous().view(-1,hidden) #[bsz, seq-1, hidden] 
        mu1 = upper_mu[:,1:,:].contiguous().view(-1,hidden)
        sigma2 = down_sigma[:, :-1, :].contiguous().view(-1, hidden)  # [bsz, seq-1, hidden] 
        sigma1 = upper_sigma[:, 1:, :].contiguous().view(-1, hidden)
        value = 0.5*(sigma2-sigma1+ (torch.exp(sigma1)+(mu1-mu2)**2)/torch.exp(sigma2) -1 ) #[bsz*(seq-1), hidden]
        return torch.mean(value,dim=1).mean()
    def sample_z(self,dec_out,enc_out,dec_enc_attn_ma,slf_attn_mask,train_mode):
        
        dec_output = dec_out.clone()
        enc_output = enc_out.clone()  
        dec_enc_attn_mask = dec_enc_attn_ma.clone()
        attention_output, _ = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        bsz,seq,hidden = attention_output.shape
        if self.flat_method=='flat':
            new_attention = torch.zeros_like(attention_output)
            for i in range(seq):
                new_attention[:, i, :] = attention_output[:, :i + 1, :].mean(dim=1)
        elif self.flat_method=='casual_attention':
            new_attention,_ = self.slf_attn_second(attention_output,attention_output,attention_output,mask=slf_attn_mask)
        elif self.flat_method=='none':
            new_attention = attention_output
        else:
            raise NotImplementedError()
        upper_mu = self.upper_mu(new_attention)  #recognition
        down_mu = self.down_mu(new_attention)  #prior
        upper_sigma = self.upper_sigma(new_attention)
        down_sigma = self.down_sigma(new_attention)
        #obtain upper mu and down mu
        #obtain KL loss
        kl_loss = self.obtain_kl(upper_mu,down_mu,upper_sigma,down_sigma)
  
        if train_mode=='train':
            z = self.sampling(upper_mu[:,1:,:],upper_sigma[:,1:,:])
            # final_z = z[:,1:,:]
            # final_z[:,0,:] = z[:,0,:]
            if self.one_step_loss:
                pri_z = self.sampling(down_mu[:,:-1,:],down_sigma[:,:-1,:])
            else:
                pri_z = None
            return z,pri_z,kl_loss
        else:
            # print(f'sample from {train_mode}')
            z = self.sampling(down_mu[:,:-1,:],down_sigma[:,:-1,:])
            return z,z,None
    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None,last_priz_predict=None,train_mode='train'):

        dec_output, dec_slf_attn = self.slf_attn( 
            dec_input, dec_input, dec_input, mask=slf_attn_mask) 

        if self.cvae:
            z,pri_z,kl_loss = self.sample_z(dec_output,enc_output,dec_enc_attn_mask,slf_attn_mask,train_mode) 
          
            dec_output[:,:-1,:] +=z 
            if train_mode!='infer' and self.one_step_loss:
                if last_priz_predict is not None:
                    pri_z = pri_z +last_priz_predict
                priz_predict = self.pri_ffn(pri_z)
            else:
                priz_predict = None

        dec_output_temp,dec_enc_attn = self.enc_attn_second(dec_output,enc_output,enc_output,mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output_temp)
        if self.second_sample=='one':
            log_var = torch.ones_like(dec_output)
            dec_output = self.sampling(dec_output,log_var)
        elif self.second_sample =='learn':
            log_var = self.pos_ffn_var(dec_output_temp)
            dec_output = self.sampling(dec_output,log_var)
        elif self.second_sample =='none':
            pass
        else:
            raise NotImplementedError()
        if self.cvae:
            return dec_output, kl_loss,priz_predict
        else:
            return dec_output
