# Cuda toolkit download

- When `nvidia-smi` is not available, you can download the toolkit from [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
- first need to install the driver, then the toolkit.
- `conda install cuda -c nvidia/label/cuda-11.3.0` if we want to install cuda 11.3.0
- Then we need to go pytorch website to install the corresponding pytorch version. `conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia/label/cuda11.3.0`