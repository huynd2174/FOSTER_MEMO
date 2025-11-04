# MEMO-FOSTER: Advanced Class-Incremental Learning with Optimized Computational Efficiency

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Abstract

**MEMO-FOSTER** represents a novel approach to class-incremental learning that synergistically combines the memory-efficient management of MEMO with the feature enhancement capabilities of FOSTER. Our methodology achieves exceptional computational efficiency while maintaining competitive performance, making it particularly suitable for cloud-based training environments and resource-constrained scenarios.

### Key Achievements

- **🎯 Balanced Performance**: Achieves 59.30% CNN accuracy, significantly outperforming MEMO (49.77%) while remaining competitive with FOSTER (62.25%)
- **⚡ Computational Efficiency**: Delivers 71.8% reduction in exemplar management time compared to FOSTER
- **🆕 Superior New Task Learning**: Attains 79.30% accuracy on new tasks, the highest among all evaluated methods
- **🧠 Optimized NME Classifier**: Achieves 56.31% accuracy, outperforming both MEMO and FOSTER baselines
- **☁️ Cloud-Optimized Architecture**: Specifically engineered for Google Colab's GPU time constraints

## Experimental Results on CIFAR-100

### Comprehensive Performance Analysis

| Method | CNN Accuracy (%) | NME Accuracy (%) | CNN Top-5 (%) | NME Top-5 (%) |
|--------|------------------|------------------|---------------|---------------|
| MEMO | 49.77 | 46.40 | 75.82 | 71.58 |
| FOSTER | 62.25 | 53.42 | 87.85 | 82.96 |
| **MEMO-FOSTER** | **59.30** | **56.31** | **85.70** | **84.51** |

## Methodology

### Research Motivation

Traditional class-incremental learning approaches face inherent limitations:

- **MEMO**: Demonstrates efficient memory management but exhibits limited overall performance
- **FOSTER**: Achieves high performance but suffers from significant computational overhead and extended processing times

### MEMO-FOSTER Architecture

Our proposed method strategically combines the strengths of both approaches:

1. **MEMO Integration**: Leverages efficient memory management and meta-optimization techniques
2. **FOSTER Enhancement**: Incorporates feature boosting and knowledge distillation capabilities
3. **Novel Optimization**: Implements optimized exemplar management with balanced performance-efficiency trade-offs

### Technical Framework

#### Feature Boosting Mechanism
- **Residual Learning**: Trains new modules to learn residuals relative to existing model parameters
- **Multi-branch Architecture**: Integrates features from both legacy and novel network branches
- **Balanced Loss Function**: Combines Cross-Entropy and Knowledge Distillation for knowledge retention

<p align="center"><img src='imgs/gradientboosting.png' width='900' alt='Gradient Boosting Visualization'></p>

#### Feature Compression Strategy
- **Knowledge Distillation**: Transfers knowledge from teacher (post-boosting model) to compact student model
- **Temperature-controlled KD**: Implements balanced class weights with configurable temperature parameters
- **Model Replacement**: Student model replaces teacher for subsequent iterations to control computational complexity

<p align="center"><img src='imgs/boosting.png' width='900' alt='Feature Boosting Architecture'></p>

#### MEMO Integration Framework
- **Shallow Layer Freezing**: Preserves general feature representations while enabling deep layer adaptation
- **Exemplar Rehearsal**: Maintains representative samples to mitigate catastrophic forgetting
- **Configurable Freeze Depth**: Supports `stage_2` for CIFAR and `layer2` for ImageNet architectures

<p align="center"><img src='imgs/compression.png' width='900' alt='Feature Compression Process'></p>

## Detailed Experimental Analysis

### Computational Efficiency Metrics

| Performance Metric | FOSTER | MEMO-FOSTER | Improvement |
|-------------------|--------|-------------|-------------|
| Exemplar Reduction Time (s) | 13.5 | **1.0** | **92.6%** |
| Exemplar Construction Time (s) | 6.7 | **4.7** | **29.9%** |
| Total Exemplar Management (s) | 20.2 | **5.7** | **71.8%** |
| Training Time per Epoch (s) | 2.0-2.3 | **1.9-2.1** | **7-9%** |
| Total Training Time (min) | ~90 | **~82** | **8.9%** |

### Incremental Learning Capability Assessment

| Method | CNN Old Tasks (%) | CNN New Task (%) | NME Old Tasks (%) | NME New Task (%) |
|--------|-------------------|------------------|-------------------|------------------|
| MEMO | 47.41 | 71.00 | 42.90 | 77.90 |
| FOSTER | 60.72 | 76.00 | 51.34 | 72.10 |
| **MEMO-FOSTER** | **57.08** | **79.30** | **55.09** | **67.30** |

### Learning Trajectory Analysis

| Task | MEMO | FOSTER | MEMO-FOSTER |
|------|------|--------|-------------|
| T0 | 91.3 | 93.6 | **93.6** |
| T1 | 78.5 | 84.95 | **82.5** |
| T2 | 74.0 | 82.13 | **79.57** |
| T3 | 68.47 | 77.38 | **74.03** |
| T4 | 63.86 | 74.6 | **70.88** |
| T5 | 61.48 | 71.62 | **68.07** |
| T6 | 57.37 | 69.44 | **66.11** |
| T7 | 54.64 | 66.04 | **62.84** |
| T8 | 51.93 | 63.54 | **60.82** |
| T9 | 49.77 | 62.25 | **59.30** |

## Installation and Configuration

### Experimental Environment

Our experiments were conducted on Google Colaboratory with the following specifications:
- **Platform**: Google Colaboratory (Colab)
- **GPU**: Tesla T4/V100 (dynamically allocated)
- **RAM**: 12.7 GB
- **Framework**: PyTorch 1.12+
- **CUDA**: 11.2+
- **Python**: 3.8+

### Dependencies Installation

```bash
pip install torch torchvision
pip install tqdm numpy
```

### Dataset Preparation

- **CIFAR-100**: Download and extract to `data/cifar-100-python/` directory
- **ImageNet-100**: Prepare train/test lists in `imagenet-sub/train.txt` and `imagenet-sub/eval.txt`

### Configuration Parameters

Essential parameters for MEMO-FOSTER implementation:

```json
{
    "model_name": "memo-foster",
    "dataset": "cifar100",
    "memory_size": 2000,
    "memory_per_class": 20,
    "init_cls": 10,
    "increment": 10,
    "convnet_type": "resnet32",
    "memo_freeze_until": "stage_2",
    "kd_temperature": 2,
    "kd_alpha": 1.0,
    "init_epochs": 200,
    "boosting_epochs": 170,
    "compression_epochs": 130,
    "lr": 0.1,
    "batch_size": 128,
    "weight_decay": 0.0005
}
```

## Usage Instructions

### Training MEMO-FOSTER on CIFAR-100

```bash
python main.py --config configs/cifar/b0inc10.json
```

### Baseline Method Execution

```bash
# FOSTER baseline
python main.py --config configs/foster-imagenet100.json

# FOSTER-RMM (optional)
python main.py --config configs/foster-rmm.json
```

**Note**: For CPU-only execution, configure `"device": [-1]` in the configuration file.

### Progress Monitoring

- Logs are systematically saved to `logs/{model_name}/{dataset}/{init_cls}/{increment}/...`
- Real-time reporting of Top-1/Top-5 accuracy for both CNN and NME classifiers
- Comprehensive loss tracking during boosting and compression phases

## Performance Visualizations

<p align="center">
<img src='imgs/performance.png' width='900' alt='Performance Comparison Charts'>
</p>

<p align="center">
<img src='imgs/vis.png' width='900' alt='Visualization Results'>
</p>

## Application Recommendations

### Optimal Use Cases

| Scenario | Rationale for MEMO-FOSTER Selection |
|----------|-------------------------------------|
| **Google Colab/Cloud Computing** | Optimized for GPU time constraints and computational efficiency |
| **Production Systems** | Balanced performance-efficiency trade-off for real-world deployment |
| **Continuous Learning Applications** | Superior new task learning while maintaining old task knowledge |
| **Resource-Constrained Environments** | Optimal resource utilization with competitive performance |
| **Scalable Production Systems** | Stable, reliable performance with growing number of tasks |

### Hyperparameter Optimization

- **`memo_freeze_until`**: Balance stability-plasticity trade-off
  - CIFAR ResNet-32: `"stage_2"`
  - ImageNet ResNet-18: `"layer2"`
- **`kd_alpha` and `kd_temperature`**: Control compression stability without performance degradation
- **`memory_size`**: Increase when model compression is effective (if computational budget allows)

## Citation

If you use MEMO-FOSTER in your research, please cite the original FOSTER paper:

```bibtex
@article{wang2022foster,
  title={FOSTER: Feature Boosting and Compression for Class-Incremental Learning},
  author={Wang, Fu-Yun and Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
  journal={arXiv preprint arXiv:2204.04662},
  year={2022}
}
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyCIL**: [GitHub Repository](https://github.com/G-U-N/PyCIL)
- **FOSTER**: [ECCV22-FOSTER Repository](https://github.com/G-U-N/ECCV22-FOSTER)
- **MEMO**: [ICLR23-MEMO Repository](https://github.com/wangkiw/ICLR23-MEMO)


**MEMO-FOSTER: Advancing Class-Incremental Learning through Computational Efficiency and Performance Optimization**
