
## Installation


Our work is based on MMAction2. Please refer to [install.md](docs/install.md) for installation.
SGDNet uses PyTorch version 1.8.0 and Python version 3.8.

## Get Started

About the basic usage of MMAction2, please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.

## Data Preparation of TITAN-Human Action

Our constructed TITAN-Human Action dataset utilized raw data from the paper [TITAN: Future Forecast using Action Priors](https://arxiv.org/abs/2003.13886) ,it published at CVPR 2020.
If you need to access the raw data, please contact the authors of [TITAN](https://usa.honda-ri.com/titan).

You can obtain the relevant files used in our paper from [here](https://pan.baidu.com/s/1R4EpwGraRI29gBuf7_jtCg?pwd=nxbw), including the embedding file, dataset annotation files, and files used for loading data.


## Model Evaluation
After preparing the dataset and obtaining the model, you can evaluate the model using the following code. 


```
python tools/test.py your_config_path/gcn_sentence_csatt.py your_model_path/best_mAP@0.5IOU_epoch_10.pth --eval mAP
```
