



## 🔧 Dataset preprocessing 

<!-- ## Download from YouTube -->

### Environment

```shell
sudo apt-get install ffmpeg
sudo apt-get install yt-dlp
pip install yt-dlp
pip install facenet-pytorch==2.6.0
pip install mmcv==2.2.0

git clone https://github.com/m-bain/whisperX.git
cd whisperX
pip install -e .

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

