import subprocess
from tqdm import tqdm
import os
import argparse
def download(type,id,save_path,no_audio):
    # save_path = r'D:\research\3D\code\face_recognition\Youtube\raw_videos'
    url = ''
    if type=='video':
        url = f"https://www.youtube.com/watch?v={id}"
    elif type=='list':
        url = f"https://www.youtube.com/playlist?list={id}"  #downloading a playlist (several videos)
    else:
        raise NotImplementedError()
    

    command = [
        "yt-dlp",
        # "--proxy",
        # "http://127.0.0.1:10809", #7890
        url,
        "--merge-output-format",
        "mp4", #mkv
        "-o",
        f"{save_path}/%(id)s.%(ext)s", 
        "--external-downloader",
        "aria2c",
        "--downloader-args",
        "aria2c:-x 16 -k 1M"
    ]
    if no_audio:
        command = [
            "yt-dlp",
            # "--proxy",
            # "http://127.0.0.1:10809",
            url,
            "-f",
            "bestvideo[ext=mp4]",
            "--merge-output-format",
            "mp4",
            "-o",
            f"{save_path}/%(id)s.%(ext)s",  
            "--external-downloader",
            "aria2c",
            "--downloader-args",
            "aria2c:-x 16 -k 1M"
        ]
    subprocess.run(command)



def process(mkv_path,mp4_path):
    '''
    '''
    os.makedirs(mp4_path,exist_ok=True)
    videos = os.listdir(mkv_path)
    new_video_ids = [name.replace('.mkv','') for name in videos]
    no_wrong_prefix = []
    for id in new_video_ids:
        if id[0]=='-':
            # id[0]
            new_id = '_'+id[1:]
        else:
            new_id = id
        no_wrong_prefix.append(new_id)
    # final_set = set(no_wrong_prefix) | set(all_ids)
    # exit()
    # remove_repu = set(videos) +set()
    for each_id in tqdm(set(no_wrong_prefix)):
        if os.path.exists(os.path.join(mkv_path,each_id+'.mkv'))==False:
            continue
        command = [
        "ffmpeg",
        "-i", os.path.join(mkv_path,each_id+'.mkv'),
        "-codec", "copy",
        "-strict", "-2",
        os.path.join(mp4_path,each_id+'.mp4')
        ]
        subprocess.run(command)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',type=str,choices=['video','list'])
    parser.add_argument('--id',type=str,required=True)
    parser.add_argument('--save_path',type=str,required=True)
    parser.add_argument('--no_audio',action='store_true')
    
    args = parser.parse_args()
    download(args.type,args.id,args.save_path,args.no_audio)