❗❗❗**Note: this work is currently under peer review. We will release all the associated datasets and codes following the receipt of the review comments. We appreciate your interest and kindly ask you to stay tuned for upcoming updates.**

## 📰 News
- **[2024-10-21]: Dataset preprocessing codes, EmoAva test set, and partial generated results are released.** 





## 🔧 Dataset preprocessing 

<!-- ## Download from YouTube -->

### Environment

```shell
sudo apt-get install ffmpeg
sudo apt-get install yt-dlp
pip install yt-dlp

git clone https://github.com/m-bain/whisperX.git
cd whisperX
pip install -e .

facenet-pytorch==2.6.0
mmcv==2.2.0
```


### Running command
```
python youtube_down.py \
--type video \
--id youtube_id \
--save_path your_path \
# --no_audio 
```


Get [token](https://huggingface.co/pyannote/speaker-diarization-3.1) and then type it in line 81 in process_youtube.py


```
python process_youtube.py \
--video_path PATH \  # videos (path) downloaded from the above file
--audio_path PATH \ # path where to save the audio files (.flac format)
--time_json PATH \ # a json file to record the timestamps
--split_time_path PATH \ # videos (path) sliced along time.
--data_json PATH \ # metadata file in json format
--single_video_path PATH  # videos (path) sliced along space.
```
For 3DMM parameters extraction, please refer to [EMOCA](https://github.com/radekd91/emoca/tree/release/EMOCA_v2/gdl_apps/EMOCA):
```
python emoca/gdl_apps/EMOCA/demos/test_emoca_on_video.py \
--input_video VIDEO_PATH \
--output_folder YOUR_PATH \
--save_codes 
```

## Dataset recipe

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



## 🗒️TODO list

- [ ] releasing the EmoAva dataset and evaluation codes
- [ ] releasing the raw videos/audio and the code for automated dataset construction (preprocessing)
- [ ] releasing the code of CTEG model
- [ ] relaxing the code of GiGA model


## 🗳️ Some generated expressions

<div style="text-align: center;">

| Text: What a beautiful story. | Text: I'm sorry, honey. | Text: so, yeah, good to see you. | Text: I am so dead. |
| :----------------------: | :----------------------: | :----------------------: | :----------------------: |
| <video src="https://github.com/user-attachments/assets/5166c81b-9945-47cd-aecb-f46068dd79df" style="width:220px; height:240px; display: block; margin: 0 auto; max-width: 100%; max-height: 100%;" /> | <video src="https://github.com/user-attachments/assets/e2884e98-c592-4796-8a90-a645126d3561" style="width:220px; height:240px;" /> | <video src="https://github.com/user-attachments/assets/549280d2-43fd-4718-a56a-c9c7cb5298e1" style="width:220px; height:240px;" /> | <video src="https://github.com/user-attachments/assets/d625c273-5c3e-4b8b-8670-52bc08932988" style="width:220px; height:240px;" /> |
| **Text: I just know that with you <br> by my side that anything is possible.** | ****Text:** What the hell?** | **Text:** **That'd be great!** | **Text:** **Yeah, it was really hard.** |
| <video src="https://github.com/user-attachments/assets/d203c84b-0fe5-45c4-9554-55671be2105f" style="width:220px; height:240px;" /> | <video src="https://github.com/user-attachments/assets/226c4d88-ca1c-4bb1-9bfe-9c03c71dd370" style="width:220px; height:240px;" /> | <video src="https://github.com/user-attachments/assets/2c7260c1-86dd-4087-8452-6208ab1ad7cc" style="width:220px; height:240px;" /> | <video src="https://github.com/user-attachments/assets/d6b21680-0b57-4283-8448-b7436f6242eb" style="width:220px; height:240px;" /> |
</div>
