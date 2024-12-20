import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    """
    Train the U-Net model
    """
    # Load and parse the configuration file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
   
    
    diffusion_params = config['diffusion_params']
    dataset_params = config['dataset_params']
    model_params = config['model_params']
    train_params = config['train_params']
    
    # Initialize the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_params['num_timesteps'],
                                     beta_start=diffusion_params['beta_start'],
                                     beta_end=diffusion_params['beta_end'])
    
    # Load the dataset
    mnist = MnistDataset('train', im_path=dataset_params['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_params['batch_size'], shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = Unet(model_params).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_params['task_name']):
        os.mkdir(train_params['task_name'])
    
    # Load checkpoint ifit exists
    if os.path.exists(os.path.join(train_params['task_name'],train_params['ckpt_name'])):
        print('Loading existing checkpoint...')
        model.load_state_dict(torch.load(os.path.join(train_params['task_name'],
                                                      train_params['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = train_params['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_params['lr'])
    criterion = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        losses = []
        for img in tqdm(mnist_loader):
            optimizer.zero_grad()
            img = img.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(img).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_params['num_timesteps'], (img.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_img = scheduler.add_noise(img, noise, t)
            noise_pred = model(noisy_img, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(train_params['task_name'],
                                                    train_params['ckpt_name']))
    
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM model training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
