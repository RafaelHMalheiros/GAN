**Introduction**
FuseDream usa GANs pré-treinados (apoiamos BigGAN-256 e BigGAN-512 por enquanto) e CLIP para obter geração de texto para imagem de alta fidelidade.

**Requirements**
Use pip ou conda para instalar os sguintes packages: PyTorch==1.7.1, torchvision==0.8.2, lpips==0.1.4 e os requerimentos de BigGAN.

Colab: https://colab.research.google.com/drive/1j6-uF0k6UdcKsutZVCpeNtPKmT7UWw2d#scrollTo=EXMSuW2EQWsd

**Citations**

@inproceedings{
brock2018large,
title={Large Scale {GAN} Training for High Fidelity Natural Image Synthesis},
author={Andrew Brock and Jeff Donahue and Karen Simonyan},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1xsqj09Fm},
}
and

@misc{
liu2021fusedream,
title={FuseDream: Training-Free Text-to-Image Generation with Improved CLIP+GAN Space Optimization}, 
author={Xingchao Liu and Chengyue Gong and Lemeng Wu and Shujian Zhang and Hao Su and Qiang Liu},
year={2021},
eprint={2112.01573},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
