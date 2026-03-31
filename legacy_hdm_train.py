#!/usr/bin/env python3
"""
Legacy HDM stage1/stage2 training and sampling entrypoint.

This keeps the original CVPR'24 HDM training code separate from the current
dual-branch ProciGen train/test path.
"""
import os
import sys

import hydra
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Pointclouds

from configs.structured import ProjectConfig
from trainer import Trainer

import training_utils

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
sys.path.append(os.getcwd())


class TrainerBehave(Trainer):
    def get_gt_pclouds(self, batch, i):
        return Pointclouds([batch["pclouds"][i].to("cuda")])

    def get_input_image(self, batch, i):
        return batch["images"][i]

    def get_seq_name(self, batch, i):
        return batch["sequence_name"][i]

    def get_seq_category(self, batch, ind=0):
        return batch["synset_id"][ind]

    def get_metadata(self, batch, i):
        metadata = dict(
            index=i,
            sequence_name=batch["sequence_name"][i],
            sequence_category=batch["synset_id"],
            frame_timestamp=batch["view_id"][i],
            camera=self.get_camera(batch, i),
            image_size_hw=batch["image_size_hw"][i],
            image_path=batch["image_path"][i],
            mask_path=batch["image_path"][i],
            center=batch["gt_trans"][i],
            radius=batch["radius"][i],
        )
        if "closest_hum" in batch:
            metadata.update(
                closest_hum=batch["closest_hum"][i],
                closest_obj=batch["closest_obj"][i],
                pred_hum=batch["pred_hum"][i],
                pred_obj=batch["pred_obj"][i],
            )
        return metadata

    def get_camera(self, batch, i):
        if self.cfg.dataset.type == "behave-objonly-segm":
            t = batch["T_obj_scaled"][i][None]
        elif self.cfg.dataset.type == "behave-humonly-segm":
            t = batch["T_hum_scaled"][i][None]
        else:
            t = batch["T"][i][None]
        return PerspectiveCameras(
            R=batch["R"][i][None],
            T=t,
            K=batch["K"][i][None],
            in_ndc=True,
            device="cuda",
        )

    def get_batch_size(self, batch):
        return len(batch["images"])


class TrainerBinarySegm(TrainerBehave):
    def compute_loss(self, batch, model):
        loss, loss_sep = model(batch, mode="train")
        self.loss_sep = loss_sep
        return loss

    def logging_addition(self, log_dict: dict):
        log_dict["train_loss_noise"] = self.loss_sep[0]
        log_dict["train_loss_mse"] = self.loss_sep[1]
        return log_dict

    def add_log_item(self, metric_logger):
        metric_logger.add_meter(
            "train_loss_noise",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        metric_logger.add_meter(
            "train_loss_mse",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        metric_logger.add_meter(
            "val_loss_noise",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        metric_logger.add_meter(
            "val_loss_mse",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        return metric_logger


class TrainerCrossAttnHO(TrainerBinarySegm):
    def logging_addition(self, log_dict: dict):
        log_dict["train_loss_hum"] = self.loss_sep[0]
        log_dict["train_loss_obj"] = self.loss_sep[1]
        return log_dict

    def add_log_item(self, metric_logger):
        metric_logger.add_meter(
            "train_loss_hum",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        metric_logger.add_meter(
            "train_loss_obj",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        metric_logger.add_meter(
            "val_loss_hum",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        metric_logger.add_meter(
            "val_loss_obj",
            training_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
        )
        return metric_logger

    def get_input_image(self, batch, i):
        return batch["images_fullcrop"][i]

    def get_gt_pclouds(self, batch, i):
        pc_h = batch["pclouds"][i] * 2 * batch["radius_hum"][i] + batch["cent_hum"][i]
        pc_o = batch["pclouds_obj"][i] * 2 * batch["radius_obj"][i] + batch["cent_obj"][i]
        return Pointclouds([torch.cat([pc_h, pc_o], 0).to("cuda")])


@hydra.main(config_path="configs", config_name="configs", version_base="1.1")
def main(cfg: ProjectConfig):
    if cfg.model.model_name == "diff-ho-attn":
        trainer = TrainerCrossAttnHO(cfg)
    elif cfg.model.predict_binary:
        trainer = TrainerBinarySegm(cfg)
    else:
        trainer = TrainerBehave(cfg)

    if cfg.run.job == "sample":
        trainer.run_sample(cfg)
    else:
        trainer.train(cfg)


if __name__ == "__main__":
    main()
