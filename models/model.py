import torch
import pytorch_lightning as pl

from .network import Network
from .utils import save_images


class Palette(pl.LightningModule):
    def __init__(
        self,
        unet_conf: dict,
        beta_schedule_conf: dict,
        init_type: str,
        gain: float,
        lr: float,
        weight_decay: float,
        sample_num: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.sample_num = sample_num

        self.loss = torch.nn.MSELoss()

        self.net = Network(unet_conf, beta_schedule_conf, init_type, gain)
        self.net.init_weights()
        self.net.set_loss(self.loss)
        self.net.set_new_noise_schedule()

    def scale_tensor(self, input_tensor):
        tensor = input_tensor.detach()[:].float().cpu()
        return (tensor + 1) / 2

    def get_current_visuals(self, gt_image, cond_image, mask, mask_image, output):
        res = {
            "gt_image": self.scale_tensor(gt_image),
            "cond_image": self.scale_tensor(cond_image),
            "mask": mask.detach()[:].float().cpu(),
            "mask_image": self.scale_tensor(mask_image),
            "output": self.scale_tensor(output),
        }
        return res

    def save_current_results(self, gt_image, mask_image, visuals, path, batch_size):
        ret_path = []
        ret_result = []
        for idx in range(batch_size):
            ret_path.append("GT_{}".format(path[idx]))
            ret_result.append(gt_image[idx].detach().float().cpu())

            ret_path.append("Process_{}".format(path[idx]))
            ret_result.append(visuals[idx :: batch_size].detach().float().cpu())

            ret_path.append("Out_{}".format(path[idx]))
            ret_result.append(visuals[idx - batch_size].detach().float().cpu())

            ret_path.append("Mask_{}".format(path[idx]))
            ret_result.append(mask_image[idx].detach().float().cpu())

        results_dict = dict(zip(ret_path, ret_result))
        return results_dict

    def training_step(self, batch, batch_idx):
        cond_image, gt_image = batch["cond_image"], batch["gt_image"]
        mask = batch["mask"]
        loss = self.net(gt_image, cond_image, mask=mask)
        return loss

    def validation_step(self, batch, batch_idx):
        cond_image, gt_image = batch["cond_image"], batch["gt_image"]
        mask, mask_image = batch["mask"], batch["mask_image"]
        path, batch_size = batch["path"], len(mask)
        output, visuals = self.net.restoration(
            cond_image,
            y_t=cond_image,
            y_0=gt_image,
            mask=mask,
            sample_num=self.sample_num,
        )

        # for key, value in self.get_current_visuals(
        #     gt_image, cond_image, mask, mask_image, output
        # ).items():
        #     self.logger.experiment.add_image(key, value)
        save_images(
            self.save_current_results(gt_image, mask_image, visuals, path, batch_size),
            self.logger.log_dir,
            self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.net.parameters())),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {"optimizer": optimizer}
