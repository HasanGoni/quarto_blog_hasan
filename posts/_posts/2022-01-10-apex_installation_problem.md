# Apex installation problem

* Apex github site says, At first clone the github repository.
* cd apex

 ```bash

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

* However when I tried to install, I could not install this 
* The problem says it shows error with status code 1
* With conda I was able to install it. However there was a problem, says fused_layer_norm_cuda is not available.
* Then I found

 ```python
torch.version.cuda is returning None
```

* Main problem is, we need to compile pytorch with the cuda version we have installed in our computer.
* So we need to find out first which cuda version is available in our computer. In ubuntu I can see it with

```bash
nvcc --version
```

* My cuda version is ``10.1``
* Therefore I need to find out the old version of pytorch which is compitable with this version. 
* The old versions are now available at here `https://pytorch.org/get-started/previous-versions/`
* For my version available pip installation will be 
`pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
`

With this problem has been solved

## Summary

* Need compatible pytorch version and cuda toolkit, for apex installation
* After that follow pip installation apex from git repo
