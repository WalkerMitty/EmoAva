from tqdm import tqdm
import os
import whisperx
import json
from ffmpy import FFmpeg
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse
from split_single import split_single_video

def split_video_with_time(input_dir,output_dir,start,end):
    '''
    input_dir: xxx.mp4
    output_dir: xxx.mp4
    start: time (seconds)
    end: end time (seconds)
    split the input dir video according to the ``start`` and ``end``, then save it to the output dir.
    '''
     # Load the video
    video = VideoFileClip(input_dir)

    # Check if the start and end times are within the duration of the video
    if end> video.duration:  
        end = video.duration

    # Trim the video
    trimmed_video = video.subclip(start, end)

    # Write the result to the output file
    trimmed_video.write_videofile(output_dir, codec="libx264", audio_codec="aac")

    # Close the video clips to release resources
    video.close()
    trimmed_video.close()



def audio2timestamp(audio_file,model,model_a,metadata,diarize_model):

    device = "cuda" 

    batch_size = 32 # reduce if low on GPU mem
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size,language='en')
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = result['segments']
    # start, end, words, text, speaker(sometimes none)
    for item in segments:
        del item['words']
    return segments



def extract(video_path: str, tmp_dir: str, ext: str):
    def getName(path):
        return os.path.basename(path).split('.')[0]
    audio_path = os.path.join(tmp_dir, '{}.{}'.format(getName(video_path), ext))
    ff = FFmpeg(inputs={video_path: None},
            outputs={audio_path: '-vn -acodec flac -loglevel quiet'})
    ff.run()

def obtain_audios(video_path,audio_path):
    all_files = os.listdir(video_path)

    os.makedirs(audio_path,exist_ok=True)
    for id in tqdm(all_files):
        id = id.replace('.mp4','')
        try:
            extract(os.path.join(video_path,id+'.mp4'),audio_path,'flac')
        except:
            print(f'{id} failed!')
            continue

def obtain_timestamps(timestamp_file,audio_path):
    model = whisperx.load_model("medium.en", 'cuda', compute_type="float16")
    model_a, metadata = whisperx.load_align_model(language_code='en', device='cuda')
    diarize_model = whisperx.DiarizationPipeline(use_auth_token='TYPE_YOUR_TOKEN_HERE', device='cuda')
    all_dict = {}
    ids_list = os.listdir(audio_path)
    for id in tqdm(ids_list):
        audio_file = os.path.join(audio_path,id)
        if os.path.exists(audio_file)==False:
            continue

        item = audio2timestamp(audio_file,model,model_a,metadata,diarize_model)
        all_dict[id] = item
    with open(timestamp_file, 'w', encoding='utf-8') as json_file:
        json.dump(all_dict, json_file, ensure_ascii=False, indent=4)


def split_videos(split_video_path,timestamps,raw_video_path):
    os.makedirs(split_video_path,exist_ok=True)
    with open(timestamps, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    ids_list = os.listdir(raw_video_path)
    for id in tqdm(ids_list):
        id = id.replace('.mp4','')
        try:
            times = data[id]
        except:
            continue
        for index,item in enumerate(times):
            cur_video = os.path.join(raw_video_path,id+'.mp4')
            # split_dir = os.path.join(split_video_path,id)
            # os.makedirs(split_dir,exist_ok=True)
            if item['start']>=item['end']:
                continue
            split_video_with_time(cur_video,os.path.join(split_video_path,f'{id}_{str(index)}.mp4'),item['start'],item['end'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type=str)
    parser.add_argument('--audio_path',type=str)
    parser.add_argument('--time_json',type=str, help='timestamps in json format')
    parser.add_argument('--split_time_path',type=str, help='video clips path')
    parser.add_argument('--data_json',type=str, help='final metadata')
    parser.add_argument('--single_video_path',type=str)

    args = parser.parse_args()
    
    # step1
    print('obtain audio file...')
    obtain_audios(args.video_path,args.audio_path) 

    #step 2
    print('obtain timestamps...')
    obtain_timestamps(args.time_json, args.audio_path)

    #step3
    print('split videos...')
    split_videos(args.split_time_path,args.time_json,args.video_path)

    #step4
    print('split single video...')
    split_single_video(args.time_json, args.data_json, args.single_video_path,args.split_time_path,temp_dir='tempdir')