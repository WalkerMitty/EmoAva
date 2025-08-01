from scipy.stats import norm, multivariate_normal
import numpy as np
from em_dataloader import load_dataset_split_batch,get_dataloader
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import argparse
import os
from joblib import Parallel, delayed


def multivariate_gaussian_probability_in_neighborhood(x, mu, sigma=0.2, threshold=0.8):  
    # Ensure x and mu are numpy arrays
    x = np.array(x)
    mu = np.array(mu)

    assert x.shape == mu.shape, "Dimension of x and mu must be the same"
    

    cov = np.diag([sigma**2] * len(mu))
    
    # Create a multivariate normal distribution
    mvn = multivariate_normal(mean=mu, cov=cov)
    
    # Calculate the CDF (cumulative distribution function) of the given point x
    cdf_value = mvn.cdf(x+threshold)
    
    # Calculate the CDF value of the point within the neighborhood
    neighborhood_prob = cdf_value - mvn.cdf(x -threshold) 
    return neighborhood_prob


def compute_accum_prob_and_count_for_instance(predict, golden, mask, eps):
    accum_prob = 0.0
    valid_count = 0.0
    for j in range(predict.shape[0]):
        if mask[j]:
            prob = multivariate_gaussian_probability_in_neighborhood(golden[j], predict[j])
            valid_count += 1
            prob = max(prob, eps)
            accum_prob += np.log2(prob)
        else:
            break
    return accum_prob, valid_count
def perplexity_fast(predict, golden,mask):

    predict = predict.cpu().numpy()
    golden = golden.cpu().numpy()
    mask = mask.cpu().numpy()
    bsz, seq, hidden = predict.shape
    assert predict.shape == golden.shape
    
    eps = 1e-10

    results = Parallel(n_jobs=16)(delayed(compute_accum_prob_and_count_for_instance)(
        predict[i], golden[i], mask[i], eps) for i in tqdm(range(bsz)))
    
    accum_prob, valid_count = zip(*results)
    accum_prob = np.array(accum_prob)
    valid_count = np.array(valid_count)

    H_i = -accum_prob / valid_count
    H = np.mean(H_i)
    
    return 2**H


def get_ppl(args):

    predict_raw = torch.load(args.para_predict)

    src_texts, tgt_exps, tgt_exps_steps = load_dataset_split_batch(args.split,-1,None)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    tgt_texts = [' '.join(['[UNK]'] * i) for i in tgt_exps_steps]
    config = {
        'tokenizer':tokenizer,
        'batch_size':256,
        'src_len':128,
        'trg_len':256,
        'shuffle':False,
    }

    test_loader = get_dataloader(src_texts, tgt_texts, tgt_exps,config)
    all_gold = []
    all_gold_mask = []
    for batch in test_loader:
        gold = batch['tgt_exp_gold']
        gold_mask = batch['tgt_gold_mask']
        all_gold.append(gold)
        all_gold_mask.append(gold_mask)
    concat_gold = torch.cat(all_gold,dim=0)
    concat_gold_mask = torch.cat(all_gold_mask,dim=0)
    ppl = perplexity_fast(predict_raw,concat_gold,concat_gold_mask)
    print(f'perplexity {ppl:.2f}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--para_predict', type=str, default='')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--tokenizer_path', type=str, default='')
    args = parser.parse_args()
    get_ppl(args)


