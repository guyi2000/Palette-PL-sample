import torch
from pytorch_lightning.cli import LightningCLI

from models.model import Palette
from models.dataset import CelebaHQDataModule


def cli_main():
    LightningCLI(Palette, CelebaHQDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    cli_main()
