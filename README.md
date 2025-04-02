# Setup
Clone the repository:
```
git clone https://github.com/babicdom/SID.git
cd SID
```
Create the environment:
```
conda create -n rine python=3.9
conda activate rine
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Store the datasets in `data/`:
* Download the `ProGAN` training & validation, and the `GAN-based`, `Deepfake`, `Low-level-vision`, and `Perceptual loss` test sets as desrcibed in https://github.com/PeterWang512/CNNDetection
* Download the `Diffusion` test data as desrcibed in https://github.com/Yuheng-Li/UniversalFakeDetect
* Download the ``Latent Diffusion Training data`` as described in https://github.com/grip-unina/DMimageDetection
* Download the ``Synthbuster`` dataset as desrcibed in https://zenodo.org/records/10066460
* Download the ``MSCOCO`` dataset https://cocodataset.org/#home

The `data/` directory should look like:
```
data
└── coco
└── latent_diffusion_trainingset
└── RAISEpng
└── synthbuster
└── train
      ├── airplane	
      │── bicycle
      |     .
└── val
      ├── airplane	
      │── bicycle
      |     .
└── test					
      ├── progan	
      │── cyclegan   	
      │── biggan
      │      .
      │── diffusion_datasets
                │── guided
                │── ldm_200
                |       .
```
