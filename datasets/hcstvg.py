import os
import json
from copy import deepcopy
import torch
import random

from tqdm import tqdm
import torch.utils.data as data
import numpy as np
from PIL import Image
import ffmpeg
import cv2
from torchvision.transforms import ToTensor, ToPILImage
from utils.bounding_box import BoxList
from .data_utils import make_hcstvg_input_clip

def video2image(video_dir):
    cap = cv2.VideoCapture(video_dir) 
    n = 1   
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   
    fps = cap.get(cv2.CAP_PROP_FPS)    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))    
    i = 0
    timeF = int(fps)     
    frames = []
    while cap.isOpened():
        ret,frame = cap.read() 

        if ret is False :
            break
        frames.append(frame)
    frames.append(frames[-1])
    cap.release() 
    return np.stack(frames).astype(np.uint8)


class HCSTVGDataset(data.Dataset):

    def __init__(self, cfg, split, transforms=None) -> None:
        super(HCSTVGDataset,self).__init__()
        assert split in ['train', 'test']
        self.cfg = cfg.clone()
        self.split = split
        self.transforms = transforms

        self.data_dir = cfg.DATA_DIR
        self.anno_dir = os.path.join(self.data_dir,'annos/hcstvg_v1')
        self.sent_file = os.path.join(self.anno_dir, f'{split}.json')  # split
        self.epsilon = 1e-10

        self.all_gt_data = self.load_data()
        self.clean_miss()
        self.vocab = None
        
        if cfg.DATA_TRUNK is not None:
            self.all_gt_data = self.all_gt_data[:cfg.DATA_TRUNK]
    
    def clean_miss(self):
        miss_name = '10__Gvp-cj3bmIY.mp4'
        for item in self.all_gt_data:
            if item['vid'] == miss_name:
                self.all_gt_data.remove(item)
                break
        
        miss_name = '1_aMYcLyh9OhU.mkv'
        for item in self.all_gt_data:
            if item['vid'] == miss_name:
                self.all_gt_data.remove(item)
                break
        
    def get_video_info(self,index):
        video_info = {}
        data_item = self.all_gt_data[index]
        video_info['height'] = data_item['height']
        video_info['width'] = data_item['width']
        return video_info

    def load_frames(self, data_item, load_video=True):
        video_name = data_item['vid']
        frame_ids = data_item['frame_ids']
        patience = 20
        
        if load_video:
            video_path = os.path.join(self.data_dir,'v1_video',video_name)
            h, w = data_item['height'], data_item['width']
            succ_flag = False
            for _ in range(patience):
                try:
                    out, _ = (
                        ffmpeg
                        .input(video_path)
                        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )
                    frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
                    # frames = video2image(video_path).reshape([-1, h, w, 3])
                    succ_flag = True
                    if succ_flag:
                        break
                except Exception:
                    print(video_name)
                    
            if not succ_flag:
                raise RuntimeError("Load Video Error")

            frames = frames[frame_ids]
            frames = [ToTensor()(frame) for frame in frames]
            frames = torch.stack(frames)
        else:
            raise NotImplementedError("Not Implement load from frames")
        
        return frames

    def __getitem__(self, index: int):
        """
        Usage:
            In training, sample a random clip from video
            In testing, chunk the video to a set of clips
        """
        video_data = deepcopy(self.all_gt_data[index]) 

        data_item = make_hcstvg_input_clip(self.cfg, self.split, video_data)
                
        frames = self.load_frames(data_item)   # T * C * H * W

        frame_ids = data_item['frame_ids']
        
        data_item['ori_actioness'] = data_item['actioness']
        if 'random_rample_t_gt_temp_bound' in video_data:
            data_item['gt_temp_bound'] = [int(np.clip(video_data['random_rample_t_gt_temp_bound']-int(video_data['frame_ids'][-1]*0.3/2), 0, video_data['frame_ids'][-1])), \
                                        int(np.clip(video_data['random_rample_t_gt_temp_bound']+int(video_data['frame_ids'][-1]*0.3/2), 0, video_data['frame_ids'][-1]))]
            temp_gt_begin= data_item['gt_temp_bound'][0]
            temp_gt_end = data_item['gt_temp_bound'][1]
            data_item['actioness'] = np.array([int(fid <= temp_gt_end and fid >= temp_gt_begin) for fid in frame_ids])
            temp_one = np.array([abs(fid - video_data['random_rample_t_gt_temp_bound'] ) for fid in frame_ids])
            data_item['ori_actioness'] = (temp_one == temp_one.min()).astype(int) ##find the only seen frame
            # data_item['ori_actioness'] = data_item['actioness']

        # load the sampled gt bounding box
        temp_gt = data_item['gt_temp_bound']
        if data_item['actioness'].max()==0:
            print(data_item['gt_temp_bound'])
            print(data_item['actioness'])
            print(frame_ids)
            data_item['actioness'] = (temp_one == temp_one.min()).astype(int) ##find the only seen frame
            print(data_item['actioness'])
        action_idx = np.where(data_item['actioness'])[0]
        start_idx, end_idx = action_idx[0], action_idx[-1]
        

        if 'track_bboxs' in video_data:
            # start_idx, end_idx = 0, len(data_item['actioness'])-1
            bbox_idx = [frame_ids[idx] for idx in range(start_idx,end_idx + 1)]
            bboxs = torch.from_numpy(video_data['track_bboxs'][bbox_idx].astype(np.int64)).reshape(-1, 4)
        else:
            bbox_idx = [frame_ids[idx] - temp_gt[0] for idx in range(start_idx,end_idx + 1)]
            bboxs = torch.from_numpy(data_item['bboxs'][bbox_idx]).reshape(-1, 4)
        

        w, h = data_item['width'], data_item['height']
        box_mask = torch.zeros(len(bboxs), h, w).to(bboxs.device).float()
        for i in range(len(bboxs)):
            box_mask[i][int(bboxs[i][1]):int(bboxs[i][3]), int(bboxs[i][0]):int(bboxs[i][2])] = 1
            
        bboxs = BoxList(bboxs, (w, h), 'xyxy')
        
        sentence = data_item['description']
        sentence = sentence.lower()
        input_dict = {'frames': frames, 'boxs': bboxs, 'text': sentence, \
                'actioness' : data_item['actioness'], 'box_mask': box_mask}

        if self.transforms is not None:
            input_dict = self.transforms(input_dict)
        
        #########################################################################################################################
        
        targets = {
            'item_id' : data_item['item_id'],
            'frame_ids' : data_item['frame_ids'],
            'actioness' : torch.from_numpy(data_item['actioness']) ,
            'ori_actioness': torch.from_numpy(data_item['ori_actioness']), 
            'start_heatmap' : torch.from_numpy(data_item['start_heatmap']),
            'end_heatmap' : torch.from_numpy(data_item['end_heatmap']),
            'boxs' : input_dict['boxs'],
            'img_size' : input_dict['frames'].shape[2:],
            'ori_size' : (h, w),
            'box_mask' : input_dict['box_mask']
        }
        
        return input_dict['frames'], sentence, targets

    def __len__(self) -> int:
        return len(self.all_gt_data)

    def load_data(self):
        """
        Prepare the Input Data Cache and the evaluation data groundtruth
        """
        cache_dir = os.path.join(self.data_dir,'data_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
         # Used for Model Input
        dataset_cache = os.path.join(cache_dir, f'hcstvg-{self.split}-input.cache')
        # Used For Evaluateion
        gt_anno_cache = os.path.join(cache_dir, f'hcstvg-{self.split}-anno.cache')
        
        if os.path.exists(dataset_cache):
            data = torch.load(dataset_cache)
            return data
        
        gt_data, gt_anno = [], []
        vstg_anno = self.preprocess(self.sent_file)
        
        for anno_id in tqdm(vstg_anno):  
            gt_file = vstg_anno[anno_id]
            frame_nums = gt_file['frame_count']
            video_name = gt_file['vid']
        
            start_fid = 0
            end_fid = frame_nums - 1
            temp_gt_begin = max(0, gt_file['tube_start_frame'])
            temp_gt_end = min(gt_file['tube_end_frame'], end_fid)

            assert len(gt_file['target_bboxs']) == temp_gt_end - temp_gt_begin + 1
            
            frame_ids = []
            for frame_id in range(start_fid, end_fid):
                frame_ids.append(frame_id)
                    
            actioness = np.array([int(fid <= temp_gt_end and fid >= temp_gt_begin) for fid in frame_ids]) 
            
            # prepare the temporal heatmap
            action_idx = np.where(actioness)[0]
            start_idx, end_idx = action_idx[0], action_idx[-1]
            
            start_heatmap = np.ones(actioness.shape) * self.epsilon
            pesudo_prob = (1 - (start_heatmap.shape[0] - 3) * self.epsilon - 0.5) / 2
            
            start_heatmap[start_idx] = 0.5
            if start_idx > 0:
                start_heatmap[start_idx-1] = pesudo_prob
            if start_idx < actioness.shape[0] - 1:
                start_heatmap[start_idx+1] = pesudo_prob

            end_heatmap = np.ones(actioness.shape) * self.epsilon
            end_heatmap[end_idx] = 0.5
            if end_idx > 0:
                end_heatmap[end_idx-1] = pesudo_prob
            if end_idx < actioness.shape[0] - 1:
                end_heatmap[end_idx+1] = pesudo_prob

            bbox_array = []
            for idx in range(len(gt_file['target_bboxs'])):
                bbox = gt_file['target_bboxs'][idx]
                x1, y1, w, h = bbox
                bbox_array.append(np.array([x1,y1,x1+w,y1+h]))
                assert x1 <= gt_file['width'] and x1 + w <= gt_file['width']
                assert y1 <= gt_file['height'] and y1 + h <= gt_file['height']
            
            bbox_array = np.array(bbox_array)
            assert bbox_array.shape[0] == temp_gt_end - temp_gt_begin + 1
            
            gt_bbox_dict = {fid : bbox_array[fid - temp_gt_begin].tolist() \
                    for fid in range(temp_gt_begin, temp_gt_end + 1)}
            
            gt_item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'bboxs' : gt_bbox_dict,
                'description' : gt_file['sentence'],
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'frame_count' : gt_file['frame_count']
            }
            
            item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'frame_ids' : frame_ids,
                'width' : gt_file['width'],
                'height' : gt_file['height'],
                'start_heatmap': start_heatmap,
                'end_heatmap': end_heatmap,
                'actioness': actioness,
                'bboxs' : bbox_array,
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'description' : gt_file['sentence'],
                'object' : 'person',
                'frame_count' : gt_file['frame_count']
            }
            
            gt_data.append(item)
            gt_anno.append(gt_item)
        
        random.shuffle(gt_data)
        torch.save(gt_data, dataset_cache)
        torch.save(gt_anno, gt_anno_cache)
        return gt_data

    def preprocess(self,anno_file):
        """
        preoprocess from the original annotation
        """
        pair_cnt = 0
        print(f"Prepare {self.split} Data")
        
        with open(anno_file, 'r') as fr:
            hcstvg_anno = json.load(fr)

        proc_hcstvg_anno = {}
        for vid in tqdm(hcstvg_anno):
            anno = hcstvg_anno[vid]
            data_pairs = {}
            data_pairs['vid'] = vid
            data_pairs['width'] = anno['width']
            data_pairs['height'] = anno['height']
            data_pairs['frame_count'] = anno['img_num']
            data_pairs['tube_start_frame'] = anno['st_frame'] - 1
            data_pairs['tube_end_frame'] = data_pairs['tube_start_frame'] + len(anno['bbox']) - 1
            data_pairs['tube_start_time'] = anno['st_time']
            data_pairs['tube_end_time'] = anno['ed_time']
            data_pairs['id'] = pair_cnt
            data_pairs['sentence'] = anno['caption']
            data_pairs['target_bboxs'] = anno['bbox']
            proc_hcstvg_anno[pair_cnt] = data_pairs
            pair_cnt += 1
        
        print(f'{self.split} pair number : {pair_cnt}')
        return proc_hcstvg_anno
