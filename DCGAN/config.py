import torch.nn as nn

class TrainingConfig():
  img_size = 64
  input_channels = 3
  latent_vector_size = 100
  gen_feature_size = 64
  dis_feature_size = 64

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( config.latent_vector_size, config.gen_feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.gen_feature_size * 8),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size*8) x 4 x 4``
            nn.ConvTranspose2d(config.gen_feature_size * 8, config.gen_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_feature_size * 4),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size*4) x 8 x 8``
            nn.ConvTranspose2d( config.gen_feature_size * 4, config.gen_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_feature_size * 2),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size*2) x 16 x 16``
            nn.ConvTranspose2d( config.gen_feature_size * 2, config.gen_feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.gen_feature_size),
            nn.ReLU(True),
            # state size. ``(config.gen_feature_size) x 32 x 32``
            nn.ConvTranspose2d( config.gen_feature_size, config.input_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)