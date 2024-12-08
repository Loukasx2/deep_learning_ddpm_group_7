import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_latest_index(output_dir):
    """
    Find the highest index of existing images in the output directory.
    """
    if not os.path.exists(output_dir):
        return 0

    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if not png_files:
        return 0

    # Extract numerical indices from filenames
    indices = [int(f.split('_')[1].split('.')[0]) for f in png_files if '_' in f and f.split('_')[1].split('.')[0].isdigit()]
    return max(indices) + 1 if indices else 0

def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    Continue from the latest sample in the directory.
    """
    # Generate random noise as the starting point
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)

    # Create output directory for saving images
    output_dir = os.path.join(train_config['task_name'], 'samples')
    os.makedirs(output_dir, exist_ok=True)

    # Get the latest index to continue sampling
    start_index = get_latest_index(output_dir)

    # Reverse diffusion process
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Predict noise for timestep `i`
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x_t-1 and x_0
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Save the final denoised images (at t=0)
    ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()  # Rescale values to [0, 1]
    ims = (ims + 1) / 2  # Normalize to [0, 1] for saving
    for idx, img in enumerate(ims):
        img_pil = torchvision.transforms.ToPILImage()(img)
        img_pil.save(os.path.join(output_dir, f"sample_{start_index + idx:05d}.png"))

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/sampling.yaml', type=str)
    args = parser.parse_args()
    infer(args)
