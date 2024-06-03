import torch.nn as nn
import torch

class TrainingConfig():
  img_size = 64
  input_channels = 3
  latent_vector_size = 100
  gen_feature_size = 64
  dis_feature_size = 64

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.latent_block = nn.Sequential(
            nn.Linear(config.latent_vector_size,1920),
            nn.ReLU(True),
        )

        self.condition_block = nn.Sequential(
            nn.Embedding(config.num_classes,128)
        )

        self.common_block = nn.Sequential(
            # state size. ``(config.gen_feature_size*8) x 2  x 2``
            nn.ConvTranspose2d(config.gen_feature_size*8, config.gen_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_feature_size * 4),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size*4) x 4 x 4``
            nn.ConvTranspose2d(config.gen_feature_size * 4, config.gen_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_feature_size * 2),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size*2) x 8 x 8``
            nn.ConvTranspose2d( config.gen_feature_size * 2, config.gen_feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_feature_size),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size) x 16 x 16``
            nn.ConvTranspose2d( config.gen_feature_size, config.input_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 32 x 32``
        )

    def forward(self, input, label):
        out1 = self.latent_block(input)
        out1 = out1.view(-1,480,2,2)

        out2 = self.condition_block(label)
        out2 = out2.view(-1,32,2,2)

        out = torch.cat((out1,out2),dim = 1)
        return self.common_block(out)