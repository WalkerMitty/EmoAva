## When Words Smile😀: Generating Diverse Emotional Facial Expressions from Text

## 📰 News
- **Dataset preprocessing codes, EmoAva test set, and partial generated results are released.** 2024-10 
- **🎉🎉🎉 All Datasets are released. The training, inference and visualization codes are released.** 2025-8
- **The paper is accepted by EMNLP 2025 (oral).**





## 📂 README Overview



- [📰 News](#-news)
- [📂 README Overview](#-readme-overview)
- [😊Dataset recipe](#dataset-recipe)
  - [How to use](#how-to-use)
  - [👀 Visualization of dataset samples](#-visualization-of-dataset-samples)
  - [Download the training set](#download-the-training-set)
  - [Preprocessing scripts](#preprocessing-scripts)
- [🧐Continuous Text-to-Expression Model](#continuous-text-to-expression-model)
  - [Environment](#environment)
  - [Training](#training)
  - [Inference](#inference)
  - [Validation](#validation)
  - [Visualization](#visualization)
- [📺Some visualization videos](#some-visualization-videos)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)




## 😊Dataset recipe

### How to use
```python
import pickle
with open('dataset/test_stage1_text.pkl', 'rb') as file:
    loaded_list_of_str = pickle.load(file)
with open('dataset/test_stage1_exps.pkl', 'rb') as file:
    loaded_list_of_exps = pickle.load(file)
print(loaded_list_of_str[:2])
print(loaded_list_of_exps[0])

#Output
# ["the two aren't mutually exclusive!", 'You should throw this out.']
# [[ 0.00248473  0.55392444  0.41335008 ...  0.02582393  0.05078617
#   -0.07435226]
#  ...
```

### 👀 Visualization of dataset samples


<table>
    <tr>
        <td><strong>1</strong></td>
        <td colspan="2"><strong>Text</strong>: The two aren't mutually exclusive!</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/S03E07_rgb2.gif" width="150" height="150" alt="video"></td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/S03E07_geometry2.gif" width="150" height="150" alt="video"></td>
    </tr>
    <tr>
        <td><strong>2</strong></td>
        <td colspan="2"><strong>Text</strong>: You should throw this out.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/dia170_rgb2.gif" width="150" height="150" alt="video"></td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/dia170_geometry2.gif" width="150" height="150" alt="video"></td>
    </tr>
    <tr>
         <td><strong>... </strong></td>
        <td colspan="5"><strong>... </strong></td>
    </tr>
        <tr>
         <td><strong>15, 000</strong></td>
        <td colspan="5"><strong>... </strong></td>
    </tr>
</table>


### Download the training set

Full expression codes and a baseline model can be found in [EmoAva](https://drive.google.com/drive/folders/1YqNRyk4QliBTpaz8hQl_4_HgQgB6zssK).

We also provide single-person, single-view video and audio files. Researchers are required to complete the license_agreement.pdf and send it to [mail](182haidong@gmail.com) to obtain access.

Please note that researchers must also obtain permission for both the [MELD](https://github.com/declare-lab/MELD) and [MEMOR](https://dl.acm.org/doi/10.1145/3394171.3413909) datasets.

### Preprocessing scripts
If you wish to construct the dataset yourself from Youtube, please refer to [README](./dataset/README.md).


## 🧐Continuous Text-to-Expression Model

### Environment
```shell
conda create -n cteg python=3.10
conda activate cteg
cd src
pip install -r requirements.txt
# (optional) If you want to visualize the results, then run:
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```

### Training

Download the dataset and put it in the `EmoAva/dataset` directory.


```
EmoAva
|-- dataset
|-- |-- train_stage1_text.pkl
|-- |-- train_stage1_exps.pkl
|-- |-- test_stage1_text.pkl
|-- |-- test_stage1_exps.pkl
|-- |-- dev_stage1_text.pkl
|-- |-- dev_stage1_exps.pkl
|-- |-- stage1_mean.npy
|-- |-- stage1_std.npy
```
Set `pretrained_path` in `src/train.sh` to the downloaded bert-base-cased model path, then run:
```shell
bash train.sh
```

### Inference

Set the following parameters in `infer.sh`
```shell
# typing your trained model here, or you can use the released model in EmoAva.
-model "xx/output/model.chkpt"
# downloaded bert-base-cased model
-tokenizer_path "bert-base-cased"
# your absolute path to the dataset
-data_source "xx/EmoAva/dataset"
```
Then run
```shell
bash infer.sh
```

### Validation

Set `infer_mode` in infer.sh to `p`, obtain the prediction result: `para_result.pt`
Then run

```shell
python evaluate.py \
	--para_predict para_result.pt \
	--split test \
	--tokenizer_path bert-base-cased
```
This script will output the continuous PPL metric. More details can be found in our paper.

### Visualization

Render facial videos from the expression vectors in the test set:

```shell
python visualize.py \
--exp_path ../dataset/test_stage1_exps.pkl \
--output all_videos
```
Note: You need to obtain `FLAME_albedo_from_BFM.npz` (~400MB) from [this](https://github.com/TimoBolkart/BFM_to_FLAME), then put it to `src/data/`.







## 📺Some visualization videos
<div style="text-align: center;">
  <table style="margin: 0 auto; border-collapse: collapse;">
    <tr>
      <td>Text: I am so dead.</td>
      <td>Text: That'd be great!</td>
      <td>Text: What a beautiful story.</td>
      <td>Text: What the hell?</td>
    </tr>
    <tr>
      <td>
        <video src="https://github.com/user-attachments/assets/d625c273-5c3e-4b8b-8670-52bc08932988" style="width:220px; height:240px;"></video>
      </td>
      <td>
        <video src="https://github.com/user-attachments/assets/2c7260c1-86dd-4087-8452-6208ab1ad7cc" style="width:220px; height:240px;"></video>
      </td>
      <td>
        <video src="https://github.com/user-attachments/assets/5166c81b-9945-47cd-aecb-f46068dd79df" style="width:220px; height:240px;"></video>
      </td>
      <td>
        <video src="https://github.com/user-attachments/assets/226c4d88-ca1c-4bb1-9bfe-9c03c71dd370" style="width:220px; height:240px;"></video>
      </td>
    </tr>
    <tr>
      <td><a href="https://github.com/user-attachments/assets/d625c273-5c3e-4b8b-8670-52bc08932988" target="_blank">link</a></td>
      <td><a href="https://github.com/user-attachments/assets/2c7260c1-86dd-4087-8452-6208ab1ad7cc" target="_blank">link</a></td>
      <td><a href="https://github.com/user-attachments/assets/5166c81b-9945-47cd-aecb-f46068dd79df" target="_blank">link</a></td>
      <td><a href="https://github.com/user-attachments/assets/226c4d88-ca1c-4bb1-9bfe-9c03c71dd370" target="_blank">link</a></td>
    </tr>
  </table>
</div>



## Acknowledgements

- [DECA](https://github.com/yfeng95/DECA)
- [Emoca](https://github.com/radekd91/emoca)
- [attention-torch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [MELD](https://github.com/declare-lab/MELD)
- [MEMOR](https://dl.acm.org/doi/10.1145/3394171.3413909)
- [LM-listener](https://github.com/sanjayss34/lm-listener)


## Citation

