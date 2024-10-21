**This work is currently under peer review. We will release all the associated datasets and code following the receipt of the review comments. We appreciate your interest and kindly ask you to stay tuned for upcoming updates.**


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


## TODO list

- [ ] releasing the EmoAva dataset and evaluation codes
- [ ] releasing the raw videos/audio and the code for automated dataset construction (preprocessing)
- [ ] releasing the code of CTEG model
- [ ] relaxing the code of GiGA model
