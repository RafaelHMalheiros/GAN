###Preparando os imports para a geração de imagens text2image
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
from fusedream_utils import FuseDreamBaseGenerator, get_G, save_image

###Inicializando o parser para ser utilizado
parser = utils.prepare_parser()
parser = utils.add_sample_parser(parser)
args = parser.parse_args()

###Diz os valores de iterações de inicialização e de otimização
INIT_ITERS = 1000
OPT_ITERS = 1000

###Inicializa uma seed aleatoria para fazer uma imagem do 0
utils.seed_rng(args.seed) 

sentence = args.text

print('Generating:', sentence)

###Escolhe entre o modelo 256 e 512
G, config = get_G(512)
generator = FuseDreamBaseGenerator(G, config, 10) 
###Gera a imagem inicial a partir do número de iterações iniciais desejadas
z_cllt, y_cllt = generator.generate_basis(sentence, init_iters=INIT_ITERS, num_basis=5)

z_cllt_save = torch.cat(z_cllt).cpu().numpy()
y_cllt_save = torch.cat(y_cllt).cpu().numpy()
###Gera a imagem inicial a partir do número de iterações de otimização desejadas
img, z, y = generator.optimize_clip_score(z_cllt, y_cllt, sentence, latent_noise=True, augment=True, opt_iters=OPT_ITERS, optimize_y=True)
###Conta o valor de CLIP loss da imagem
score = generator.measureAugCLIP(z, y, sentence, augment=True, num_samples=20)
print('AugCLIP score:', score)
import os
if not os.path.exists('./samples'):
    os.mkdir('./samples')
save_image(img, 'samples/fusedream_%s_seed_%d_score_%.4f.png'%(sentence, args.seed, score))

