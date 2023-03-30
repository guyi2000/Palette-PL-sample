# Palette: image-to-image diffusion model

A minimum exemplar of the [Palette](https://iterative-refinement.github.io/palette/) model implemented using the [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) framework.

## Environment Setup

Setup the environment with the following commands:

```shell
conda create -n palette
conda activate palette
# Install pytorch & torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Install pytorch-lightning
conda install pytorch-lightning -c conda-forge
# Install pytorch-lightning CLO
pip install "pytorch-lightning[extra]"
# Install tensorboard
conda install tensorboard
# Install opencv for python
pip install opencv-python
```

Then, you need download the dataset [CelebaHQ here](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256). Put it in the `./data` directory and unzip.

```shell
unzip achieve.zip
```

## Usage

To **train** the model, run the following command:

```shell
python main.py fit -c ./conf/config.yaml
```

To **test** the model, run the following command:

```shell
python main.py test -c ./conf/config.yaml
```

## Acknowledgments

This code borrows heavily from [guided diffusion model](https://github.com/openai/guided-diffusion) and so on.
