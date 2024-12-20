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

def sample(model, scheduler, training_params, model_params, diffusion_params):
    r"""
    Perform sampling step by step by reversing the diffusion process.
    Save individual reconstructed images (x_0 at t=0).
    
    """
    # Generate random noise as the starting point
    noise_sample = torch.randn((training_params['num_samples'],
                      model_params['im_channels'],
                      model_params['im_size'],
                      model_params['im_size'])).to(device)
    
    # Create output directory for saving images
    output_dir = os.path.join(training_params['task_name'], 'samples')
    os.makedirs(output_dir, exist_ok=True)
    
    # Reverse diffusion process
    for step in tqdm(reversed(range(diffusion_params['num_timesteps']))):
        # Predict noise for timestep `i`
        predicted_noise = model(noise_sample, torch.as_tensor(step).unsqueeze(0).to(device))
        
        # Use scheduler to get x_t-1 and x_0
        noise_sample, reconstructed_img = scheduler.sample_prev_timestep(noise_sample, predicted_noise, torch.as_tensor(step).to(device))
    
    # Normalize and save reconstructed images (at t=0)
    final_img = torch.clamp(reconstructed_img, -1., 1.).detach().cpu()  # Rescale values to [0, 1]
    final_img = (final_img + 1) / 2  
    for idx, img in enumerate(final_img):
        img_pil = torchvision.transforms.ToPILImage()(img)
        img_pil.save(os.path.join(output_dir, f"sample_{idx:05d}.png"))

def inference(args):
    """
    Inference function to perform image generation.
    
    """
    # Load configuration from the provided YAML file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    # Extract configuration parameters
    diffusion_params = config['diffusion_params']
    model_params = config['model_params']
    training_params = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_params).to(device)
    model.load_state_dict(torch.load(os.path.join(training_params['task_name'],
                                                  training_params['ckpt_name']), map_location=device))
    model.eval()
    
    # Initialize the linear noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_params['num_timesteps'],
                                     beta_start=diffusion_params['beta_start'],
                                     beta_end=diffusion_params['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, training_params, model_params, diffusion_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    inference(args)
