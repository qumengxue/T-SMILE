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
|  |  data_cache
|  |  |_ ...
```

Prepare this structure with the following steps:

**HC-STVG**
* Download the version 1 of HC-STVG videos and annotations from [HC-STVG](https://github.com/tzhhhh123/HC-STVG). Then put it into `data/hc-stvg/v1_video` and `data/hc-stvg/annos/hcstvg_v1`.
* We provide the dataset cache for HC-STVG at [here](https://github.com/qumengxue/T-SMILE/releases/download/data_cache/hcstvg_data_cache.zip). You can download it and unzip it, and then put them into `data/hc-stvg/data_cache`. 

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

## Trained Model Weight

| Dataset | resolution | url | m_vIoU/vIoU@0.3/vIoU@0.5 | size |
|:----:|:-----:|:-----:|:-----:|:-----:|
| HC-STVG | 224 | [Model](https://cowtransfer.com/s/3788e859439640)  | 30.24/51.38/23.10 |3.1GB |


