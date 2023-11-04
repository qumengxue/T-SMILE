# T-SMILE
## Dataset Preparation
The used datasets are placed in `data` folder with the following structure.
```
data
|_ hc-stvg
|  |_ v1_video
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ annos
|  |  |_ hcstvg_v1
|  |  |  |_ train.json
|  |  |  |_ test.json
|  |_ data_cache
|  |  |_ ...
|_ vidstg
|  |_ videos
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ vstg_annos
|  |  |_ train.json
|  |  |_ ...
|  |_ sent_annos
|  |  |_ train_annotations.json
|  |  |_ ...
|  |_ data_cache
|  |  |_ ...
|_ activity
|  |_ videos
|  |  |_ [video name 0].avi
|  |  |_ [video name 1].avi
|  |  |_ ...
|  |_ data_cache
|  |  |_ ...
```

Prepare this structure with the following steps:

**HC-STVG**
* Download the version 1 of HC-STVG videos and annotations from [HC-STVG](https://github.com/tzhhhh123/HC-STVG). Then put it into `data/hc-stvg/v1_video` and `data/hc-stvg/annos/hcstvg_v1`.
* We provide the dataset cache with single frame annotation for HC-STVG at [here](https://github.com/qumengxue/T-SMILE/releases/download/data_cache/hcstvg_data_cache.zip). You can download it and unzip it, and then put them into `data/hc-stvg/data_cache`.

**VidSTG**
* Download the video for VidSTG from the [VidOR](https://xdshang.github.io/docs/vidor.html) and put it into `data/vidstg/videos`. The original video download url given by the VidOR dataset provider is broken. You can download the VidSTG videos from [this](https://disk.pku.edu.cn:443/link/5AB0927F723BB3BF80FC6DCABADAF364).
* Download the text and temporal annotations from [VidSTG Repo](https://github.com/Guaranteer/VidSTG-Dataset) and put it into `data/vidstg/sent_annos`.
* Download the bounding-box annotations from [here](https://disk.pku.edu.cn:443/link/50AA3A33DDE632F32DFD402CEAF80A2B) and put it into `data/vidstg/vstg_annos`.
* We provide the dataset cache with single frame annotation for VidSTG at [here](http://box.jd.com/sharedInfo/3F47E385EDBB56AEEEC946AFBC5707A7) (pwd:5wvxcv). You can download it and put it into `data/vidstg/data_cache`.

**ActivityNet**
* Follow the steps in [ActivityNet]([https://xdshang.github.io/docs/vidor.html](http://activity-net.org/download.html)) and download the video for ActivityNet, put it into `data/activity/videos`. 
* We provide the dataset cache with single frame annotation for ActivityNet at [here](http://box.jd.com/sharedInfo/5026E51900E66720EEC946AFBC5707A7) (pwd:b396vw). You can download it and put it into `data/activity/data_cache`.

## Setup

### Requirements

The code is tested with PyTorch 1.10.0. The other versions may be compatible as well. You can install the requirements with the following commands:

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
Then, download [FFMPEG 4.1.9](https://ffmpeg.org/download.html) and add it to the `PATH` environment variable for loading the video.

### Training
Coming soon.

### Evaluation
To evaluate the trained T-SMILE models, please run the following scripts:

# HC-STVG
```
python3 -m torch.distributed.launch \
 --nproc_per_node=4 \
 test_net.py \
 --config-file "experiments/HC-STVG/e2e_STCAT_R101_HCSTVG.yaml" \
 --use-seed \
 MODEL.WEIGHT data/hc-stvg/checkpoints/output/hcstvg_res224.pth \
 OUTPUT_DIR data/hc-stvg/checkpoints/output \
 INPUT.RESOLUTION 224
```

# VidSTG
```
python3 -m torch.distributed.launch \
 --nproc_per_node=4 \
 test_net.py \
 --config-file "experiments/VidSTG/e2e_STCAT_R101_VidSTG.yaml" \
 --use-seed \
 MODEL.WEIGHT data/vidstg/checkpoints/output/vidstg_res448.pth \
 OUTPUT_DIR data/vidstg/checkpoints/output \
 INPUT.RESOLUTION 448
```

# ActivityNet
```
python3 -m torch.distributed.launch \
 --nproc_per_node=4 \
 test_net.py \
 --config-file "experiments/Activity/e2e_STCAT_R101_Activity.yaml" \
 --use-seed \
 MODEL.WEIGHT data/activity/checkpoints/output/activity_res320.pth \
 OUTPUT_DIR data/activity/checkpoints/output \
 INPUT.RESOLUTION 320
```

## Trained Model Weight
Here are our trained checkpoints with ResNet-101 backbone for evaluation.

| Dataset | resolution | url | m_vIoU/vIoU@0.3/vIoU@0.5 | size |
|:----:|:-----:|:-----:|:-----:|:-----:|
| HC-STVG | 224 | [Model](https://cowtransfer.com/s/3788e859439640)  | 30.24/51.38/23.10 |3.1GB |

| Dataset | resolution | url | Declarative (m_vIoU/vIoU@0.3/vIoU@0.5) | Interrogative (m_vIoU/vIoU@0.3/vIoU@0.5) | size |
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| VidSTG | 448 | [Model](http://box.jd.com/sharedInfo/F04EDD7FDE10E508EEC946AFBC5707A7) (pwd:cx4db7)  | 24.07/31.26/19.46 | 20.58/25.84/15.69 |3.1GB |

| Dataset | resolution | url | m_vIoU/vIoU@0.3/vIoU@0.5 | size |
|:----:|:-----:|:-----:|:-----:|:-----:|
| ActivityNet | 320 | [Model](http://box.jd.com/sharedInfo/26C777E82D088DF3EEC946AFBC5707A7) (pwd:8ebt25)  | 17.07/20.49/7.01 |3.1GB |

## Acknowledgement
This repo is build based on [STCAT](https://github.com/jy0205/STCAT/tree/main).
