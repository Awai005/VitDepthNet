# VitDepthNet
Vision-Transformer-Depth-Net (ViTDepthNet) for Depth Estimation

## Introduction
Welcome to the ViTDepthNet repository! This model is designed to enhance depth estimation capabilities by integrating the unique strengths of Vision Transformers (ViT) with convolutional neural networks (CNNs). It's optimized for complex indoor scenes and applicable to various fields such as augmented reality and autonomous driving.

## Features
- **Hybrid Architecture**: Combines the local feature extraction capabilities of CNNs with the global contextual understanding of ViTs.
- **Enhanced Accuracy**: Tailored for high-precision depth estimation on datasets like NYUv2.
- **Application Versatility**: Ideal for a range of real-world applications from robotics to augmented reality.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourgithub/ViTDepthNet.git
cd ViTDepthNet
pip install -r requirements.txt
```

## Repository Structure
models/: Contains definitions of the hybrid transformer and CNN architectures.
data/: Scripts for data preprocessing and loading.
utils/: Helper functions and utilities for image processing.
tests/: Scripts for module testing and validation.

## Acknowledgements
This project incorporates elements from several open-source projects. We acknowledge and thank the authors and contributors of these repositories for their invaluable work:
- [DefocusNet](https://github.com/dvl-tum/defocus-net): Pioneering work in depth estimation from defocus.
- [UnsupervisedDepthFromFocus](https://github.com/unsupervised-depth-from-focus-team/udff): Techniques and insights in unsupervised learning for depth estimation.
- [DeepLabV3+](https://github.com/tensorflow/models/tree/master/research/deeplab): Advanced segmentation techniques adapted for depth estimation tasks.
- [DEReD](https://github.com/Ehzoahis/DEReD): For specific methods used in self-supervised depth estimation from defocus clues.
## Code Availability
The full codebase will be made publicly available upon the publication of the associated paper to ensure the integrity and citation of the research work. Stay tuned for updates following the official release of our research findings.
