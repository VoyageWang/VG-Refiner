<div align="center">
    <h1><b>VG-Refiner: Towards Tool-Refined Referring Grounded Reasoning via Agentic Reinforcement Learning</b></h1>

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg?logo=arXiv)](https://arxiv.org/)
[![Data](https://img.shields.io/badge/ModelScope-Data-orange.svg)](https://modelscope.cn/datasets/VoyageWang/VGRefiner_train)
[![Model](https://img.shields.io/badge/ModelScope-Model-orange.svg)](https://modelscope.cn/models/VoyageWang/VGRefiner-7B)

[Yuji Wang](https://voyagewang.github.io/), [Wenlong Liu](https://nicehuster.github.io/), [Jingxuan Niu](), [Haoji Zhang](https://zhang9302002.github.io/), [Yansong Tang](https://andytang15.github.io/)

</div>

## 📰 News<a name="news"></a>

- 🎉 **We have released our code and paper!** Check out our [arXiv paper](https://arxiv.org/) and [Code](https://github.com/) for more details.

## Overview<a name="overview"></a>

<div align="center">
  <img src="./assests/pipline.PNG" width="90%">
</div>

- We identify a key limitation of existing TiVR methods in REC tasks and propose **VG-Refiner**, a two-stage think–rethink framework equipped with a refinement reward that enhances both reasoning and correction while preserving trust in reliable tool feedback.

- We define the TrRGR reasoning paradigm and propose two novel metrics for assessing the refinement capability of all LVLMs, and establish a unified PiTER protocol to ensure fair comparison.

- Preserving the general capabilities of the pre-trained model, our model outperforms SOTA methods on RefCOCO/+/g in accuracy and surpasses the Qwen2.5-VL-32B model in refinement ability, while trained on a small amount of task-specific data.

## Table of Contents
- [News](#news)
- [Overview](#overview)
- [Introduction](#introduction)
- [Installation](#installation)
- [Training](#training)
- [Benchmark](#benchmark)
- [Inference](#inference)
- [Citation](#citation)

## 📢 Introduction<a name="introduction"></a>

Tool-integrated visual reasoning (TiVR) has demonstrated great potential in enhancing multimodal problem-solving. However, existing TiVR paradigms mainly focus on integrating various visual tools through reinforcement learning, while neglecting to design effective response mechanisms for handling unreliable or erroneous tool outputs. This limitation is particularly pronounced in referring and grounding tasks, where inaccurate detection tool predictions often mislead TiVR models into generating hallucinated reasoning. To address this issue, we propose the VG-Refiner, the first framework aiming at the tool-refined referring grounded reasoning. Technically, we introduce a two-stage think–rethink mechanism that enables the model to explicitly analyze and respond to tool feedback, along with a refinement reward that encourages effective correction in response to poor tool results. In addition, we propose two new metrics and establish fair evaluation protocols to systematically measure the refinement ability of current models. We adopt a small amount of task-specific data to enhance the refinement capability of VG-Refiner, achieving a significant improvement in accuracy and correction ability on referring and reasoning grounding benchmarks while preserving the general capabilities of the pretrained model.

## 🔧 Installation<a name="installation"></a>

We use CUDA 12.4, Python 3.10, PyTorch 2.6.0, vllm 0.8.2, and verl 0.3.2.

```bash
conda create -n vg-refiner python==3.10 -y
conda activate vg-refiner

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install deepspeed==0.15.4

pip install vllm==0.8.2 

cd src/rl 

# It is recommended to manually install flash-attention first 
# We use flas

pip install -e .  # Install EasyR1 env
```

## 🚀 Training<a name="training"></a>

### 🔥 1. GRPO Training  

> [!NOTE]
> The recommanded training requirement for 7B model is a 4x80G GPUs server.

Training Data: [VGRefiner_train-9K](https://modelscope.cn/datasets/VoyageWang/VGRefiner_train) and  [RefCOCOG-val-evf](https://modelscope.cn/datasets/VoyageWang/VGRefiner_refcocog_val)   

Download pretrained models using the following scripts:
```bash
mkdir pretrained_models
cd pretrained_models
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

Start training using this script:
```bash
cd src/rl

bash examples/qwen2_5_vl_7b_grounding_toolcall.sh
```

### 🔄 2. Merge Checkpoint in Hugging Face Format<a name="merge-checkpoint"></a>

```bash
cd src/rl

python3 scripts/model_merger.py --local_dir your_save_path/step_xx/actor  
```

## 📊 Benchmark<a name="benchmark"></a>

Benchmark Data [VGRefiner_bench_evf_test](https://modelscope.cn/datasets/VoyageWang/VGRefiner_bench_evf_test) and [VGRefiner_bench_uninext_test](https://modelscope.cn/datasets/VoyageWang/VGRefiner_bench_uninext_test)

Model weights [VG-Refiner-7B](https://modelscope.cn/models/VoyageWang/VGRefiner-7B) and palce it in model_weights dict.


```bash
cd src/evaluation    

bash src/evaluation/eval_grounding_precomputed_tool.sh
```

## 🔍 Inference<a name="inference"></a>

```bash
cd src
python inference.py
```

## 📋 License<a name="license"></a>
![](https://img.shields.io/badge/License-MIT-blue.svg#id=wZ1Hr&originHeight=20&originWidth=82&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
This project adheres to [MIT License](https://lbesson.mit-license.org/).

## 🔖 Citation<a name="citation"></a>

If you use our work, please cite our paper.

```
@article{vg-refiner,
  title={VG-Refiner: Towards Tool-Refined Referring Grounded Reasoning via Agentic Reinforcement Learning},
  author={Yuji Wang and Wenlong Liu and Jingxuan Niu and Haoji Zhang and Yansong Tang},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## Acknowledgement
We would like to thank the following repos for their great work: 

- This work is built upon the [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl).
- This work borrow some code from [Seg-Zero](https://github.com/dvlab-research/Seg-Zero), [UniVG-R1](https://github.com/AMAP-ML/UniVG-R1), and [Dianjin-OCR](https://github.com/aliyun/qwen-dianjin/tree/master/DianJin-OCR-R1).
- This work utilizes models from [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [EVF-SAM](https://github.com/hustvl/EVF-SAM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT). 

