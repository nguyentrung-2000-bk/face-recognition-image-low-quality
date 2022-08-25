# AdaFace: Quality Adaptive Margin for Face Recognition

Official github repository for AdaFace: Quality Adaptive Margin for Face Recognition. 
The paper (https://arxiv.org/abs/2204.00964) is presented in CVPR 2022 (Oral). 



# Installation

```
conda create --name adaface pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
conda activate adaface
conda install scikit-image matplotlib pandas scikit-learn 
pip install -r requirements.txt
```

# Dataset (MS1MV2)
1. Download MS1M-ArcFace (85K ids/5.8M images) from [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) and unzip at DATASET_ROOT
2. Unpack mxrecord files to imgs with the following code.
```
python convert.py --rec_path <DATASET_ROOT>/faces_emore
```

# Train
```
# training small model (resnet18) on a subset of MS1MV2 dataset
python main.py \
    --data_root <DATASET_ROOT> \
    --train_data_path faces_emore/imgs \
    --val_data_path faces_emore \
    --train_data_subset \
    --prefix run_ir18_ms1mv2_subset \
    --gpus 2 \
    --use_16bit \
    --batch_size 512 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2
```

# Pretrained Models


| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R18  | CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing) |
| R18  | VGGFace2     | [gdrive](https://drive.google.com/file/d/1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv/view?usp=sharing) |
| R18  | WebFace4M     | [gdrive](https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing) |
| R50  | CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing) |
| R50  | WebFace4M     | [gdrive](https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing) |
| R50  | MS1MV2     | [gdrive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) |
| R100 | MS1MV2     | [gdrive](https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view?usp=sharing) |
| R100 | MS1MV3     | [gdrive](https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing) |
| R100 | WebFace4M  | [gdrive](https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing) |
| R100 | WebFace12M | [gdrive](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing) |



# Comparison Pretrain model with Finetune model

Pretrain model is AdaFace modle WebFace12M R100.


| Model    | Dataset       | F1 Score |
|----------|---------------|----------|
| Pretrain | VN Celeb      | 88,61%   |
| Finetune | VN Celeb      | 86,69%   |
| Pretrain | VN Celeb blur | 85,27%   |
| Finetune | VN Celeb blur | 91,55%   |



