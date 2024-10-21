import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from tqdm import tqdm
from PIL import Image
import mmcv,cv2
import numpy as np
import shutil
import stat
import json
import argparse

device = 'cuda'


mtcnn = MTCNN(keep_all=True,device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def crop_boxes_from_image(image, boxes):

    cropped_images = []

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_images.append(cropped_image)

    return cropped_images

def frames_to_video(input_list, output_video_path, fps=25):

    max_height = 0
    max_width = 0
    for frame_path in input_list:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        height, width, _ = frame.shape
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width


    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (max_width, max_height))


    for frame_path in input_list:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        height, width, _ = frame.shape
        top = (max_height - height) // 2
        bottom = max_height - height - top
        left = (max_width - width) // 2
        right = max_width - width - left
        color = [0, 0, 0]  
        padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        video_writer.write(padded_frame)


    video_writer.release()

    print("video has been saved to:", output_video_path)


def expand_boxes(boxes, frame_image,scale=1.2):

    expanded_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min

        new_width = width * scale
        new_height = height * scale

        center_x = x_min + width / 2
        center_y = y_min + height / 2

        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2


        expanded_boxes.append([
            max(0, new_x_min), max(0, new_y_min),
            min(frame_image.width, new_x_max), min(frame_image.height, new_y_max)
        ])

    return np.array(expanded_boxes, dtype=int)


def speaker_clip(video_path,new_video_path,folder_path):

    # folder_path = 'temp_videos'
    if os.path.exists(folder_path):
        
        shutil.rmtree(folder_path,onerror=handle_remove_readonly)
    os.makedirs(folder_path,exist_ok=True)



    # compress_video = mmcv.VideoReader(compress_path)
    origin_video = mmcv.VideoReader(video_path)

    origin_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in origin_video]
    # compress_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in compress_video]
    boxes_all = []
    each_frame_is_one = True
    print('detecting...')
    #batch_boxes or boxes_all:  list of tensors each tensor: [N, 3, size, size], N faces number
    for id,frame in enumerate(tqdm(origin_frames)):

        boxes, _, = mtcnn.detect(frame, landmarks=False)
        if boxes is None:
            continue
        faces = mtcnn.extract(frame, boxes,save_path=None)

        boxes = expand_boxes(boxes,frame,1.5)
        cropped_images = crop_boxes_from_image(origin_frames[id], boxes)
        if len(cropped_images)!=1:
            each_frame_is_one = False
            # if os.path.exists(compress_path):
                # os.remove(compress_path)
            shutil.rmtree(folder_path,onerror=handle_remove_readonly)
            return each_frame_is_one
            

        cropped_images[0].save(f'{folder_path}/{id}_face.png')  #

        boxes_all.append(faces)



    results = []
    for id,frame in enumerate(tqdm(origin_frames)):
        # target = f'./temp_videos/{id}_face'
        target=f'{folder_path}/{id}_face.png'
        results.append(target)

    frames_to_video(results,new_video_path,fps=origin_video.fps)

    shutil.rmtree(folder_path,onerror=handle_remove_readonly)
    return each_frame_is_one


def split_single_video(timestamps,final_json,output_video_path,split_path,temp_dir):

    os.makedirs(output_video_path,exist_ok=True)

    with open(timestamps, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    final_data = {}
    failed = []
    all_ids = os.listdir(split_path)
    for id in tqdm(all_ids):
        id = id.replace('.mp4','')
        item = data[id]
        cur_item = []
        for index,segment in enumerate(item):
            video_path = os.path.join(split_path,id+f'_{index}.mp4')
            new_video_path = os.path.join(output_video_path,id+f'_{index}.mp4')
            if os.path.exists(new_video_path):
                continue
            try:
                keep = speaker_clip(video_path,new_video_path,temp_dir)
            except:
                failed.append(video_path)
                continue
            if keep:
                cur_item.append({'text':segment['text'],'video_path':new_video_path})
        final_data[id] = cur_item

    print('failed ids',failed)
    with open(final_json, 'w', encoding='utf-8') as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type=str)
    parser.add_argument('--audio_path',type=str)
    parser.add_argument('--time_json',type=str, help='timestamps in json format')
    parser.add_argument('--split_time_path',type=str, help='video clips path')

    timestamps = '/data/xhd/3D/dataset/Youtube_output/timestamps_forth.json'  #no second
    # final_data = '/data/xhd/3D/dataset/Youtube_output/final_data_100_final.json'
    final_data = None
    final_videos = '/data/xhd/3D/dataset/Youtube_output/video_split_single_forth' #no second
    video_split_time = '/data/xhd/3D/dataset/Youtube_output/video_split_time_forth' #no second
    temp_dir = 'tempdir5'

    split_single_video(timestamps,final_data,final_videos,video_split_time,temp_dir)