**Note: this work is currently under peer review. We will release all the associated datasets and code following the receipt of the review comments. We appreciate your interest and kindly ask you to stay tuned for upcoming updates.**

## Some generated expressions

<table>
    <tr>
        <td><strong>Text:</strong> What a beautiful story.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_100_1029.gif" width="150" height="172.5" alt="video"></td>
        <td><strong>Text:</strong> I'm sorry, honey.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_13_832.gif" width="150" height="172.5" alt="video"></td>
    </tr>
    <tr>
        <td><strong>Text:</strong> so, yeah, good to see you.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_87_1168.gif" width="150" height="172.5" alt="video"></td>
        <td><strong>Text:</strong> I am so dead..</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_13_3.gif" width="150" height="172.5" alt="video"></td>
    </tr>
    <tr>
        <td><strong>Text:</strong> I just know that with you <br>by my side that anything is possible.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_13_925.gif" width="150" height="172.5" alt="video"></td>
        <td><strong>Text:</strong> What the hell? </td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_21_718.gif" width="150" height="172.5" alt="video"></td>
    </tr>
    <tr>
        <td><strong>Text:</strong> That'd be great!.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_13_904.gif" width="150" height="172.5" alt="video"></td>
        <td><strong>Text:</strong> Yeah, it was really hard.</td>
        <td><img src="https://github.com/WalkerMitty/EmoAva/blob/main/resource/15_s_13_74.gif" width="150" height="172.5" alt="video"></td>
    </tr>
</table>

## Dataset preprocessing 

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

### Visualization of dataset samples


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



## TODO list

- [ ] releasing the EmoAva dataset and evaluation codes
- [ ] releasing the raw videos/audio and the code for automated dataset construction (preprocessing)
- [ ] releasing the code of CTEG model
- [ ] relaxing the code of GiGA model
