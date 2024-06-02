import torch
from config import TrainingConfig,Generator
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
import argparse
from PIL import Image


config = TrainingConfig()

def find_opt_grid(x):
    # Initialize variables to store the optimal pair and the minimum difference
    optimal_a, optimal_b = None, None
    min_diff = float('inf')

    # Iterate over possible factors
    for a in range(1, int(x**0.5) + 1):
        if x % a == 0:
            b = x // a
            diff = abs(b - a)
            if diff < min_diff:
                min_diff = diff
                optimal_a, optimal_b = a, b

    return optimal_a, optimal_b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random cat images using a pre-trained DCGAN model.")
    parser.add_argument('--checkpoint_file_location', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--num_images', type=int, required=True, help='Number of images to generate.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the generated images.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the generation on (default: cpu).')

    args = parser.parse_args()

    a,b = find_opt_grid(args.num_images)

    gen_model = torch.load(args.checkpoint_file_location).to(args.device)
    gen_model.eval()

    input = torch.randn((args.num_images,config.latent_vector_size,1,1),device = args.device)
    out = gen_model(input)

    grid = vutils.make_grid(out.detach().cpu(), nrow=b, padding=2, normalize=True, scale_each=True)
    grid_np = grid.permute(1, 2, 0).mul(255).byte().numpy()

    image = Image.fromarray(grid_np)
    image.save(os.path.join(args.save_dir, 'sample.jpg'))

    print('Image saved')