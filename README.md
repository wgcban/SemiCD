## SS-CD: Semi-Supervised Change Detection in Remote Sensing Images via Consistency Regularization

#### [Paper]()

This repocitory contains the official implementation of our paper:  Revisiting Consistency Regularization for Semi-supervised Change Detection in Remote Sensing Images.

<p align="center"><img src="./imgs/method.jpg" width="900"></p>

### Requirements

This repo was tested with `Ubuntu 18.04.3 LTS`, `Python 3.7`, `PyTorch 1.1.0`, and `CUDA 10.0`. But it should be runnable with recent PyTorch versions >=1.1.0.

The required packages are `pytorch` and `torchvision`, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress.

```bash
pip install -r requirements.txt
```

### Dataset
## LEVIR-CD

## WHU-CD


### Training

To train a model, first download PASCAL VOC as detailed above, then set `data_dir` to the dataset path in the config file in `configs/config.json` and set the rest of the parameters, like the number of GPUs, cope size, data augmentation ... etc ,you can also change CCT hyperparameters if you wish, more details below. Then simply run:

```bash
python train.py --config configs/config.json
```

The log files and the `.pth` checkpoints will be saved in `saved\EXP_NAME`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```

To resume training using a saved `.pth` model:

```bash
python train.py --config configs/config.json --resume saved/CCT/checkpoint.pth
```

**Results**: The results will be saved in `saved` as an html file, containing the validation results,
and the name it will take is `experim_name` specified in `configs/config.json`.

### Pseudo-labels

If you want to use image level labels to train the auxiliary labels as explained in section 3.3 of the paper. First generate the pseudo-labels
using the code in `pseudo_labels`:


```bash
cd pseudo_labels
python run.py --voc12_root DATA_PATH
```

`DATA_PATH` must point to the folder containing `JPEGImages` in Pascal Voc dataset. The results will be
saved in `pseudo_labels/result/pseudo_labels` as PNG files, the flag `use_weak_labels` needs to be set to True in the config file, and
then we can train the model as detailed above.


### Inference

For inference, we need a pretrained model, the jpg images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --config config.json --model best_model.pth --images images_folder
```

Here are the flags available for inference:

```
--images       Folder containing the jpg images to segment.
--model        Path to the trained pth model.
--config       The config file used for training the model.
```

### Pre-trained models

Pre-trained models can be downloaded [here](https://github.com/yassouali/CCT/releases).

### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```

```

For any questions, please contact Yassine Ouali.

#### Config file details ⚙️

Bellow we detail the CCT parameters that can be controlled in the config file `configs/config.json`, the rest of the parameters
are self-explanatory.

```javascript
{
    "name": "CCT",                              
    "experim_name": "CCT",                             // The name the results will take (html and the folder in /saved)
    "n_gpu": 1,                                             // Number of GPUs
    "n_labeled_examples": 1000,                             // Number of labeled examples (choices are 60, 100, 200, 
                                                            // 300, 500, 800, 1000, 1464, and the splits are in dataloaders/voc_splits)
    "diff_lrs": true,
    "ramp_up": 0.1,                                         // The unsupervised loss will be slowly scaled up in the first 10% of Training time
    "unsupervised_w": 30,                                   // Weighting of the unsupervised loss
    "ignore_index": 255,
    "lr_scheduler": "Poly",
    "use_weak_labels": false,                               // If the pseudo-labels were generated, we can use them to train the aux. decoders
    "weakly_loss_w": 0.4,                                   // Weighting of the weakly-supervised loss
    "pretrained": true,

    "model":{
        "supervised": true,                                  // Supervised setting (training only on the labeled examples)
        "semi": false,                                       // Semi-supervised setting
        "supervised_w": 1,                                   // Weighting of the supervised loss

        "sup_loss": "CE",                                    // supervised loss, choices are CE and ab-CE = ["CE", "ABCE"]
        "un_loss": "MSE",                                    // unsupervised loss, choices are CE and KL-divergence = ["MSE", "KL"]

        "softmax_temp": 1,
        "aux_constraint": false,                             // Pair-wise loss (sup. mat.)
        "aux_constraint_w": 1,
        "confidence_masking": false,                         // Confidence masking (sup. mat.)
        "confidence_th": 0.5,

        "drop": 6,                                           // Number of DropOut decoders
        "drop_rate": 0.5,                                    // Dropout probability
        "spatial": true,
    
        "cutout": 6,                                         // Number of G-Cutout decoders
        "erase": 0.4,                                        // We drop 40% of the area
    
        "vat": 2,                                            // Number of I-VAT decoders
        "xi": 1e-6,                                          // VAT parameters
        "eps": 2.0,

        "context_masking": 2,                               // Number of Con-Msk decoders
        "object_masking": 2,                                // Number of Obj-Msk decoders
        "feature_drop": 6,                                  // Number of F-Drop decoders

        "feature_noise": 6,                                 // Number of F-Noise decoders
        "uniform_range": 0.3                                // The range of the noise
    },
```

#### Acknowledgements

- This code is based on [CCT](https://github.com/yassouali/CCT).
- Code structure was based on [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m)
- ResNet backbone was downloaded from [torchcv](https://github.com/donnyyou/torchcv)
