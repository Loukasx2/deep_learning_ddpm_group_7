# deep_learning_ddpm_group_7
The project is to re-implement the Denoising Diffusion Probabilistic Model (DDPM) in PyTorch and reproduce their results at least on MNIST and ideally on CIFAR-10.

* ```python -m tools.train``` for training ddpm
* ```python -m tools.sample``` for generating images

## Configuration
* ```config/default.yaml``` - Allows you to play with different components of ddpm  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```experiment_name``` key in config will be created

During training of DDPM the following output will be saved 
* Latest Model checkpoint in ```experiment_name``` directory

During sampling the following output will be saved
* Sampled images at the final timestep of the reverse process, thus reconstructed images $x_0$ ```experiment_name/samples/*.png```
