<p align="center">
  <h1 align="center">
    SAM-I2V++:
    <br>
    <span style="font-size: 0.8em;">
      Efficiently Upgrading SAM for Promptable Video Segmentation
    </span>
  </h1>
  <p align="center" style="font-size: 1.3em; color: #1f77b4;">
    <b>IEEE TPAMI 2026</b>
  </p>
</p>

<p align="center">
  <a href="https://mhaiyang.github.io/">Haiyang Mei</a>&nbsp;&nbsp;&nbsp;
  <a href="https://pengyuzhang.me/">Pengyu Zhang</a>&nbsp;&nbsp;&nbsp;
  <a href="https://sites.google.com/view/showlab">Mike Zheng Shou</a><sup>✉️</sup>  
  <br>
  Show Lab, National University of Singapore
</p>

<div align="center">
  <p>
    <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34" target="_blank">
      <img src="https://img.shields.io/badge/IEEE%20Xplore-Paper-grey?logo=ieee&logoColor=white&labelColor=blue">
    </a>
    &nbsp;&nbsp;
    <a href="#5-citation">
      <img src="https://img.shields.io/badge/Google%20Scholar-BibTeX-grey?logo=google-scholar&logoColor=white&labelColor=blue">
    </a>
  </p>
</div>

- [Table of Contents](#0-SAM-I2VPP)
  * [1. Overview](#1-overview)
  * [2. Installation](#2-installation)
  * [3. Getting Started](#3-getting-started)
    + [3.1 Download Checkpoint](#31-download-checkpoint)
    + [3.2 Demo Use](#32-demo-use)
    + [3.3 Testing](#33-testing)
    + [3.4 Evaluation](#34-evaluation)
    + [3.5 Training](#35-training)
    + [3.6 Web Annotation Tool](#36-web-annotation-tool)
  * [4. Acknowledgements](#4-acknowledgements)
  * [5. Citation](#5-citation)
  * [6. License](#6-license)
  * [7. Contact](#7-contact)

### 1. Overview

**SAM-I2V++** is an enhanced version of **[SAM-I2V (CVPR 2025)](https://github.com/showlab/SAM-I2V)**,
a training-efficient method that upgrades the image-based SAM for promptable video segmentation.
It achieves over **93%** of **SAM 2.1**’s performance while requiring only **0.2%** of its training cost.

<p align="center">
  <img src="assets/teaser.png?raw=true" width="400"/>
</p>

**SAM-I2V++** takes an input video and extracts frame features via an image encoder enhanced by a temporal feature integrator to capture dynamic context. These features are processed by a memory selective associator and memory prompt generator to manage historical information and generate target prompts. A prompt encoder incorporates optional user inputs (e.g., masks, points, boxes). Finally, the mask decoder produces segmentation masks for each frame, enabling user-guided and memory-conditioned promptable video segmentation.

<p align="center">
  <img src="assets/pipeline.png?raw=true" width="750"/>
</p>

### 2. Installation

Our implementation uses `python==3.11`, `torch==2.5.0` and `torchvision==0.20.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM-I2VPP on a GPU machine using:

```bash
git clone https://github.com/showlab/SAM-I2VPP.git && cd SAM-I2VPP
conda create -n sam-i2vpp python=3.11
conda activate sam-i2vpp
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### 3. Getting Started

#### 3.1 Download Checkpoint

First, we need to download the SAM-I2VPP checkpoint. It can be downloaded from:

- sam-i2vpp_8gpu.pt 
[ [Google Drive](https://drive.google.com/drive/folders/1oN16mGGndbX7a7Gj5wWWyFDN9QHxMMoU?usp=sharing) ]
[ [OneDrive](https://1drv.ms/f/c/f6d9d790b8550d3f/IgC1qxQ-jAg0RJ9dopj3TnX3ARPwJol7D68-Jo4TPuPYwSo?e=Ge4pEA) ]
[ [BaiduDisk](https://pan.baidu.com/s/15-ipIqjfrz0Qm5iOEh4R_g?pwd=pami) ]
- sam-i2vpp_32gpu.pt
[ [Google Drive](https://drive.google.com/drive/folders/1oN16mGGndbX7a7Gj5wWWyFDN9QHxMMoU?usp=sharing) ]
[ [OneDrive](https://1drv.ms/f/c/f6d9d790b8550d3f/IgC1qxQ-jAg0RJ9dopj3TnX3ARPwJol7D68-Jo4TPuPYwSo?e=Ge4pEA) ]
[ [BaiduDisk](https://pan.baidu.com/s/15-ipIqjfrz0Qm5iOEh4R_g?pwd=pami) ]

**Both models were trained in 26 hours using 24GB GPUs.** The first model (sam-i2vpp_8gpu.pt) was trained with 8 GPUs, while the second model (sam-i2vpp_32gpu.pt) was trained with 32 GPUs and offers better performance.

#### 3.2 Demo Use

SAM-I2V++ can be used in a few lines as follows for promptable video segmentation. Below provides a video predictor with APIs for example to add prompts and propagate masklets throughout a video. Same as SAM2, SAM-I2V++ supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from i2vpp.build_i2vpp import build_i2vpp_video_predictor

checkpoint = "./checkpoints/sam-i2vpp_32gpu.pt"
model_cfg = "./i2vpp/configs/i2vpp-infer.yaml"
predictor = build_i2vpp_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
  state = predictor.init_state( < your_video >)

  # add new prompts and instantly get the output on the same frame
  frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, < your_prompts >):

  # propagate the prompts to get masklets throughout the video
  for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    ...
```

#### 3.3 Testing

We provide instructions for testing on the SAV-Test dataset.

(a) Please refer to the [sav_dataset/README.md](sav_dataset/README.md) for detailed instructions on how to download and prepare the SAV-Test dataset before testing.

(b) Prepare the 'mask_info' for the ease of testing via:
```
python tools/save_gt_mask_multiprocess.py
```
Or you can directly download the preprocessed 'mask_info' [here](https://drive.google.com/drive/folders/1oN16mGGndbX7a7Gj5wWWyFDN9QHxMMoU?usp=sharing).

(c) Run the inference script
```
cd test_pvs
sh semi_infer.sh
```

#### 3.4 Evaluation

Run the evaluation script
```
sh semi_eval.sh
```

#### 3.5 Training

(a) Please refer to the [sav_dataset/README.md](sav_dataset/README.md) for detailed instructions on how to download and prepare the SAV-Train dataset. Totally 50,583 training videos ([train/txt/sav_train_list.txt](train/txt/sav_train_list.txt)).

(b) We follow SAM 2 to train the model on mixed video and image data. Download the [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/) and sample a subset of images, as the full dataset is too large to use in its entirety. We randomly sample 10k images ([train/txt/sa1b_10k_train_list.txt](train/txt/sa1b_10k_train_list.txt)) to train SAM-I2V.

(c) Download the SAM 1 model (i.e., [TinySAM](https://huggingface.co/xinghaochen/tinysam/tree/main)) to be upgraded and put it to `checkpoints/tinysam.pth`.

(d) Train the model:

- Single node with 8 GPUs:
```
sh train.sh
```

- Multi-node with each node has 8 GPUs (e.g., 4x8=32 GPUs):
```
sh multi_node_train_4_nodes.sh
```

#### 3.6 Web Annotation Tool
```
ssh -L 5000:127.0.0.1:5000 username@serverip
python tools/web_annotation_tool.py
```

### 4. Acknowledgements

Our implementation builds upon [SAM 2](https://github.com/facebookresearch/sam2) and reuses essential modules from its official codebase.

### 5. Citation

If you use SAM-I2V++ in your research, please use the following BibTeX entry.

```bibtex
@InProceedings{Mei_2025_CVPR,
    author    = {Mei, Haiyang and Zhang, Pengyu and Shou, Mike Zheng},
    title     = {SAM-I2V: Upgrading SAM to Support Promptable Video Segmentation with Less than 0.2% Training Cost},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3417-3426}
}

@article{Mei_2026_TPAMI,
    author    = {Mei, Haiyang and Zhang, Pengyu and Shou, Mike Zheng},
    title     = {SAM-I2V++: Efficiently Upgrading SAM for Promptable Video Segmentation},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    year      = {2026},
}
```

### 6. License
Please see `LICENSE`.

### 7. Contact
E-Mail: Haiyang Mei (haiyang.mei@outlook.com)


**[⬆ back to top](#1-overview)**
