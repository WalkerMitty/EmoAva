import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tgt_exps,config):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tgt_exps = tgt_exps
        self.tokenizer = config["tokenizer"]
        self.max_length_src = config["src_len"]
        self.max_length_tgt = config["trg_len"]
        if 'trg_exp_dim' in config:
            self.trg_exp_dim = config["trg_exp_dim"]
        else:
            self.trg_exp_dim = 53

    def __len__(self):
        return len(self.src_texts)

    def get_subsequent_mask(self,seq):
        ''' For masking out the subsequent info. '''

        len_s = seq.size()[0]
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def modify_tensor(self,tensor):

        true_indices = (tensor == True).nonzero(as_tuple=True)[0]

        if len(true_indices) > 0:
            first_true_index = true_indices[0]
            last_true_index = true_indices[-1]

            tensor[first_true_index] = False
            tensor[last_true_index] = False

        return tensor

    def pad_or_truncate(self,arr, max_length):
        seq, dim = arr.shape

        padded_arr = np.zeros((max_length, self.trg_exp_dim))

        if seq >= max_length:
            padded_arr[1:max_length] = arr[:max_length - 1]
        else:
            padded_arr[1:seq + 1] = arr

        tensor = torch.tensor(padded_arr, dtype=torch.float32)

        return tensor

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]

        tgt_text = self.tgt_texts[idx]
        tgt_exp = self.tgt_exps[idx] #numpy

        src_encodings = self.tokenizer(
            src_text, max_length=self.max_length_src, padding='max_length', truncation=True, return_tensors="pt"
        )
        tgt_encodings = self.tokenizer(
            tgt_text, max_length=self.max_length_tgt, padding='max_length', truncation=True, return_tensors="pt"
        )
        #original:
        tgt_attention_mask = tgt_encodings['attention_mask'].bool().squeeze()[:-1].unsqueeze(-2) & self.get_subsequent_mask(tgt_encodings['input_ids'].squeeze()[:-1])
        tgt_attention_mask = tgt_attention_mask.squeeze()
        return {
            'src_input_ids': src_encodings['input_ids'].squeeze(),
            'src_attention_mask': src_encodings['attention_mask'].bool().squeeze().unsqueeze(-2),
            'tgt_input_ids': tgt_encodings['input_ids'].squeeze()[:-1],
            'tgt_attention_mask': tgt_attention_mask,
            'tgt_exp_mask': self.modify_tensor(tgt_encodings['attention_mask'].bool().squeeze()[:-1]),
            'tgt_exp':self.pad_or_truncate(tgt_exp,self.max_length_tgt)[:-1,:],
            'tgt_exp_gold': self.pad_or_truncate(tgt_exp,self.max_length_tgt)[1:,:], #last n-, pad
            'tgt_gold_mask':tgt_encodings['attention_mask'].bool().squeeze()[1:] #[seq_len-2]
        }


def collate_fn(batch):
    src_input_ids = torch.stack([item['src_input_ids'] for item in batch])
    src_attention_mask = torch.stack([item['src_attention_mask'] for item in batch])
    tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch])
    tgt_attention_mask = torch.stack([item['tgt_attention_mask'] for item in batch])
    tgt_exp_mask = torch.stack([item['tgt_exp_mask'] for item in batch])
    tgt_exp = torch.stack([item['tgt_exp'] for item in batch])
    tgt_exp_gold = torch.stack([item['tgt_exp_gold'] for item in batch])
    tgt_gold_mask = torch.stack([item['tgt_gold_mask'] for item in batch])


    return {
        'src_input_ids': src_input_ids,
        'src_attention_mask': src_attention_mask,
        'tgt_input_ids': tgt_input_ids,
        'tgt_attention_mask': tgt_attention_mask,
        'tgt_exp_mask':tgt_exp_mask,
        'tgt_exp':tgt_exp,
        'tgt_exp_gold':tgt_exp_gold,
        'tgt_gold_mask':tgt_gold_mask
    }


def get_dataloader(src_texts, tgt_texts, tgt_exp,config):
    dataset = TranslationDataset(src_texts, tgt_texts, tgt_exp,config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn,shuffle=config['shuffle'])
    return dataloader



def load_dataset_split_batch(split,final=-1,baseline=None):
    '''
    return: List[str], List[array], List[int].
    '''
    '''
    fast mode
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(script_dir.replace('src','dataset'),f'{split}_stage1_text.pkl'), 'rb') as file:
        src_texts = pickle.load(file)
    train_path = os.path.join(script_dir.replace('src','dataset'),f'{split}_stage1_exps.pkl')


    with open(train_path, 'rb') as f:
        tgt_exps_origin = pickle.load(f)
    tgt_exps = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mean_path = os.path.join(script_dir.replace('src','dataset'), 'stage1_mean.npy')

    mean = np.load(mean_path)
    for i in range(len(tgt_exps_origin)):
        tgt_exps.append(np.concatenate((mean.reshape(1,53),tgt_exps_origin[i]),axis=0))
    tgt_exps_steps = [x.shape[0] for x in tgt_exps]

    if final!=-1:
        src_texts = src_texts[:final]
        tgt_exps = tgt_exps[:final]
        tgt_exps_steps = tgt_exps_steps[:final]
    if baseline is not None:
        if baseline=='shuffle': 
            np.random.seed(13)
            shuffled_exps = []
            for i in range(len(tgt_exps)):
                seq = tgt_exps[i].shape[0]
                shuffled_array = tgt_exps[i][np.random.permutation(seq)]
                shuffled_exps.append(shuffled_array)
            return src_texts,shuffled_exps,tgt_exps_steps
        elif baseline=='random':  

            combined_list = list(zip(tgt_exps, tgt_exps_steps))
 
            random.shuffle(combined_list)

 
            shuffled_list1, shuffled_list2 = zip(*combined_list)

            shuffled_exps = list(shuffled_list1)
            shuffled_steps = list(shuffled_list2)
            return src_texts,shuffled_exps,shuffled_steps
        else:
            raise NotImplementedError()
    return src_texts,tgt_exps,tgt_exps_steps








