import argparse
import cv2

import numpy as np

import json
import os
import cv2
from tqdm import tqdm
import torch
import pickle
import sys
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets

import argparse
def gen_image(deca, codedict, include_im, fix_cam=True):
    if fix_cam:
        codedict['cam'][0,0] = 5.
        codedict['cam'][0,1] = 0.
        codedict['cam'][0,2] = 0.05
 
    opdict, visdict = deca.decode(codedict) # , include_im=include_im) #tensor
    landmarks = {'landmarks2d': visdict['landmarks2d']}  #1,3,224,224



    if include_im:
        #remainder = {'inputs': visdict['inputs'], 'shape_detail_images': visdict['shape_detail_images']}
        remainder = {'shape_detail_images': visdict['shape_detail_images'], 'inputs': visdict['inputs']}
    else:
        remainder = {'shape_images': visdict['shape_images']}   


    return deca.visualize(remainder, size=640), deca.visualize(landmarks, size=640)



def make_a_video_gt(args):
    '''
    '''
    if os.path.exists(args.output) == False:
        os.mkdir(args.output)

    with open(args.exp_path, 'rb') as file:
        exps = pickle.load(file)

    deca = DECA(config = deca_cfg, device='cuda')

    with open(args.default_code, 'rb') as file:
        default = pickle.load(file)
    
    bsz = len(exps)
    # seq = exps.size()[1]
    # seq = 128
    zeros = torch.zeros(3).to('cuda')
    for i in tqdm(range(bsz)):
        first_exps = exps[i]
        seq = first_exps.shape[0]
        first_exps = torch.tensor(first_exps).to('cuda')
        frames_list = []
        for j in tqdm(range(seq)):
            exp = first_exps[j][:50]
            jaw = first_exps[j][50:]
            final_jaw = torch.concat((zeros,jaw))

            gt_code = {
                'exp': exp.cuda().view(1,-1),
                'pose': final_jaw.cuda().view(1,-1)
            }

            for key in default:
                if key not in {'exp', 'pose'}:
                    gt_code[key] = default[key].float().cuda()

            gt_image, _ = gen_image(deca, gt_code, include_im=False)
            frames_list.append(gt_image)

        fps = 24.0  #24.0

        frame_size = (frames_list[0].shape[1], frames_list[0].shape[0])


        out = cv2.VideoWriter(os.path.join(args.output,f'{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

        for frame in frames_list:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        # exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default='test_stage1_exps.pkl')
    parser.add_argument('--default_code', type=str, default='./data/default_code_trevor_emoca2.pkl')
    parser.add_argument('--output', type=str, default='./')
    args = parser.parse_args()
    make_a_video_gt(args)