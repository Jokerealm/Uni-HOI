"""
Trainer for training and testing

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import datetime
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional
import os.path as osp

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TVF

import training_utils
import diffusion_utils
from dataset import get_dataset
from configs.structured import ProjectConfig
from accelerate import DistributedDataParallelKwargs
from pytorch3d.structures import Pointclouds
import numpy as np
import pickle as pkl

from eval.chamfer_distance import chamfer_distance
torch.multiprocessing.set_sharing_strategy('file_system') # fix some bug in some servers

class Trainer(object):
    def __init__(self, cfg:ProjectConfig):
        # Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # fix ddp problem
        accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu,
                                  gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
                                  kwargs_handlers=[ddp_kwargs])

        # Logging
        training_utils.setup_distributed_print(accelerator.is_main_process)
        if cfg.run.job == 'sample':
            cfg.logging.wandb = False
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.init(project=cfg.logging.wandb_project, name=cfg.run.name, job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
            wandb.run.log_code(root=hydra.utils.get_original_cwd(),
                               include_fn=lambda p: any(
                                   p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
                               exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
            cfg: ProjectConfig = DictConfig(wandb.config.as_dict())  # get the configs back from wandb for hyperparameter sweeps

        # Configuration
        # print(OmegaConf.to_yaml(cfg))
        print(f'Current working directory: {os.getcwd()}')

        # Set random seed
        training_utils.set_seed(cfg.run.seed)

        # Initialize model
        from model import get_model
        model = get_model(cfg)

        # Exponential moving average of model parameters
        if cfg.ema.use_ema:
            from torch_ema import ExponentialMovingAverage
            model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
            model_ema.to(accelerator.device)
            print('Initialized model EMA')
        else:
            model_ema = None
            print('Not using model EMA')
        self.model_ema = model_ema

        # Optimizer and scheduler
        optimizer = training_utils.get_optimizer(cfg, model, accelerator)
        scheduler = training_utils.get_scheduler(cfg, optimizer)

        # Resume from checkpoint and create the initial training state
        self.train_state: training_utils.TrainState = self.load_checkpoint(cfg, model, model_ema, optimizer, scheduler)

        # Datasets
        from dataset import get_dataset
        dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

        # Compute total training batch size
        self.total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

        # Setup. Note that this does not currently work with CO3D.
        model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis = accelerator.prepare(
            model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis)

        # for later use
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.dataloader_train, self.dataloader_val, self.dataloader_vis = dataloader_train, dataloader_val, dataloader_vis
        self.cfg = cfg
        self.accelerator = accelerator

        # additional data buffer
        self.loss_sep = None

    def load_checkpoint(self, cfg, model, model_ema, optimizer, scheduler):
        "load optimizer, model state, scheduler etc. "
        return training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler,
                                                     model_ema)

    def cleanup_gpu_memory(self):
        """清理GPU显存"""
        import gc
        
        print("Clearing GPU cache...")
        
        # 清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # 强制垃圾回收
        gc.collect()
        
        # 打印显存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
        
        print("GPU memory cleanup completed")

    def train(self, cfg:ProjectConfig):
        fscore_last, chamf_last = 0, 0.
        # Visualize before training
        if cfg.run.job == 'vis' or cfg.run.vis_before_training:
            fscores, chamfs = self.visualize(
                cfg=cfg,
                model=self.model,
                dataloader_vis=self.dataloader_vis,
                accelerator=self.accelerator,
                identifier=f'{self.train_state.step}',
                num_batches=1,
            )
            print(f"F-score={np.mean(fscores):.4f}, chamf={np.mean(chamfs):.4f}")
            fscore_last, chamf_last = np.mean(fscores), np.mean(chamfs)
            if cfg.run.job == 'vis':
                if cfg.logging.wandb and self.accelerator.is_main_process:
                    wandb.finish()
                    time.sleep(5)
                return

        self.print_info(cfg)

        # prepare for training
        train_state, optimizer, scheduler = self.train_state, self.optimizer, self.scheduler
        model, model_ema = self.model, self.model_ema
        accelerator = self.accelerator
        dataloader_train, dataloader_val, dataloader_vis = self.dataloader_train, self.dataloader_val, self.dataloader_vis


        # training loop
        while True:
            # Train progress bar
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('fscore', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('chamf', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger = self.add_log_item(metric_logger)
            progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq,
                                                                  header=log_header)

            # Train
            for i, batch in enumerate(progress_bar):
                if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
                model.train()

                # Gradient accumulation
                with accelerator.accumulate(model):

                    # Forward
                    loss = self.compute_loss(batch, model)

                    # Backward
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if cfg.optimizer.clip_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                        grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                    # Step optimizer
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        scheduler.step()
                        train_state.step += 1

                    # Exit if loss was NaN
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(90)

                # Gradient accumulation
                if accelerator.sync_gradients:
                    # Logging
                    log_dict = {
                        'lr': optimizer.param_groups[0]["lr"],
                        'step': train_state.step,
                        'train_loss': loss_value,
                        'grad_norm_clipped': grad_norm_clipped,
                        "fscore": fscore_last,
                        "chamf": chamf_last
                    }
                    log_dict = self.logging_addition(log_dict)
                    metric_logger.update(**log_dict)
                    if (
                            cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                        wandb.log(log_dict, step=train_state.step)

                    # Update EMA
                    if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                        model_ema.update(model.parameters())

                    # Save a checkpoint
                    if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                        self.save_checkpoint(accelerator, cfg, model, model_ema, optimizer, scheduler, train_state)

                    # Visualize
                    if (cfg.run.vis_freq > 0) and (train_state.step % cfg.run.vis_freq) == 0: # 5k steps
                        fscores, chamfs = self.visualize(
                            cfg=cfg,
                            model=model,
                            dataloader_vis=dataloader_vis,
                            accelerator=accelerator,
                            identifier=f'{train_state.step}',
                            num_batches=1,
                        )

                        fscore_last, chamf_last = np.mean(fscores), np.mean(chamfs)
                        print(f"updated F-score={fscore_last:.4f}, chamf={chamf_last:.4f}")

                    # End training after the desired number of steps/epochs
                    # or when lr is decreased to zero
                    if train_state.step >= cfg.run.max_steps or optimizer.param_groups[0]['lr'] < 1e-8:
                        print(f'Ending training at: {datetime.datetime.now()}')
                        print(f'Final train state: {train_state}')

                        # 清理显存
                        print('Cleaning up GPU memory...')
                        self.cleanup_gpu_memory()

                        wandb.finish()
                        time.sleep(5)
                        return

            # Epoch complete, log it and continue training
            train_state.epoch += 1

            # Gather stats from all processes
            metric_logger.synchronize_between_processes(device=self.accelerator.device)
            print(f'{log_header}  Average stats --', metric_logger)

    def print_info(self, cfg):
        # Info
        print(f'***** Starting training at {datetime.datetime.now()} *****')
        print(f'    Dataset train size: {len(self.dataloader_train.dataset):_}')
        print(f'    Dataset val size: {len(self.dataloader_train.dataset):_}')
        print(f'    Dataloader train size: {len(self.dataloader_train):_}')
        print(f'    Dataloader val size: {len(self.dataloader_val):_}')
        print(f'    Batch size per device = {cfg.dataloader.batch_size}')
        print(f'    Total train batch size (w. parallel, dist & accum) = {self.total_batch_size}')
        print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
        print(f'    Max training steps = {cfg.run.max_steps}')
        print(f'    Training state = {self.train_state}')

    def save_checkpoint(self, accelerator, cfg, model, model_ema, optimizer, scheduler, train_state):
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        checkpoint_dict = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': train_state.epoch,
            'step': train_state.step,
            'best_val': train_state.best_val,
            'model_ema': model_ema.state_dict() if model_ema else {},
            'cfg': cfg
        }
        checkpoint_path = 'checkpoint-latest.pth'
        # check if checkpoint exist
        if osp.isfile(checkpoint_path):
            ckpt_old = torch.load(checkpoint_path, weights_only=False)
            if train_state.step > 60000 and train_state.step %5000 ==0 :
                # keep old checkpoints that have been trained for more than 100k steps \
                old_step = ckpt_old['step']
                newfile = checkpoint_path.replace('-latest.pth', f'-step-{old_step:07d}.pth')
                torch.save(ckpt_old, newfile)

            accelerator.save(checkpoint_dict, checkpoint_path)
            print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

        else:
            accelerator.save(checkpoint_dict, checkpoint_path)
            print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

    def compute_loss(self, batch, model):
        return model(batch, mode='train')

    def run_sample(self, cfg:ProjectConfig):
        # Whether or not to use EMA parameters for sampling
        if cfg.run.sample_from_ema:
            assert self.model_ema is not None
            self.model_ema.to(self.accelerator.device)
            sample_context = self.model_ema.average_parameters
        else:
            sample_context = nullcontext
        # Sample
        with sample_context():
            self.sample(
                cfg=cfg,
                model=self.model,
                dataloader=self.dataloader_val,
                accelerator=self.accelerator,
                output_dir=cfg.run.save_name
            )
        if cfg.logging.wandb and self.accelerator.is_main_process:
            wandb.finish()
        print('all done')
        time.sleep(5)

    def logging_addition(self, log_dict:dict):
        return log_dict

    def add_log_item(self, metric_logger):
        return metric_logger

    def is_done(self, batch, output_dir: str):
        "check if this batch is done"
        bs = self.get_batch_size(batch)
        filename = '{name}.{ext}'
        filestr = str(output_dir / '{dir}' / '{category}' / filename)
        for i in range(bs):
            pred_file = filestr.format(dir='pred', category=self.get_seq_category(batch, i), name=self.get_seq_name(batch, i), ext='ply')
            if not os.path.isfile(pred_file):
                return False
        return True

    @torch.no_grad()
    def sample(self, cfg: ProjectConfig,
                model: torch.nn.Module,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        from pytorch3d.io import IO
        from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
        from pytorch3d.structures import Pointclouds
        from tqdm import tqdm

        # Eval mode
        model.eval()
        progress_bar: Iterable[FrameData] = tqdm(dataloader, disable=(not accelerator.is_main_process))

        # Output dir
        output_dir: Path = Path(output_dir)

        # PyTorch3D IO
        # io = IO()

        end_idx = cfg.run.batch_end if cfg.run.batch_end is not None else len(dataloader)
        # Visualize
        for batch_idx, batch in enumerate(progress_bar):
            progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d}')
            if cfg.run.num_sample_batches is not None and batch_idx >= cfg.run.num_sample_batches:
                break

            # only run for specific batches
            if cfg.dataset.type == 'shapenet_r2n2':
                if batch_idx < cfg.run.batch_start:
                    print(f"Skipped batch {batch_idx}.")
                    continue
                if batch_idx >= end_idx:
                    break

            # import pdb;
            # pdb.set_trace() # batch keys:
            # print([k for k in batch])

            # for debug: save sampled frames
            filename = '{name}.{ext}'
            filestr = str(output_dir / '{dir}' / '{category}' / filename)
            sequence_category = self.get_seq_category(batch, 0) # TODO: replace for different dataset

            file = filestr.format(dir='images', category=sequence_category, name=f"batch_{batch_idx:02d}", ext='json')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            json.dump(batch['image_path'], open(file, 'w'))
            print("sequence:", sequence_category, 'first image:', batch['image_path'][0])
            # continue

            # Optionally produce multiple samples for each point cloud
            for sample_idx in range(cfg.run.num_samples):
                if self.is_done(batch, output_dir) and not cfg.run.redo:
                    print(f"batch {batch_idx} already done, skipped")
                    continue

                # Filestring
                filename = f'{{name}}-{sample_idx}.{{ext}}' if cfg.run.num_samples > 1 else '{name}.{ext}'
                filestr = str(output_dir / '{dir}' / '{category}' / filename)

                # Sample
                w_joint = 0 if cfg.model.model_name != 'diff-comb' else cfg.model.model_joint_weight
                w_sep = 0 if cfg.model.model_name != 'diff-comb' else cfg.model.model_sep_weight
                output, all_outputs = model(batch, mode=cfg.run.sample_mode,
                                            return_sample_every_n_steps=10,
                                            scheduler=cfg.run.diffusion_scheduler,
                                            num_inference_steps=cfg.run.num_inference_steps,
                                            disable_tqdm=(not accelerator.is_main_process),
                                            noise_step=cfg.run.sample_noise_step,
                                            w_joint=w_joint, w_sep=w_sep, # for combined diffusion model
                                            eta=cfg.model.ddim_eta,
                                            )
                output: Pointclouds
                all_outputs: List[Pointclouds]  # list of B Pointclouds, each with a batch size of return_sample_every_n_steps

                # Save individual samples
                for i in range(len(output)):
                    sequence_name = self.get_seq_name(batch, i)
                    sequence_category = self.get_seq_category(batch, i)

                    (output_dir / 'gt' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'pred' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'images' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'metadata' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'evolutions' / sequence_category).mkdir(exist_ok=True, parents=True)

                    # Save ground truth
                    self.save_pclouds(batch, filestr, i, output, sequence_category, sequence_name, cfg.run.sample_save_gt)

                    # Save input images
                    filename = filestr.format(dir='images', category=sequence_category, name=sequence_name, ext='png')
                    TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)
                    # self.save_input_image(batch, filename, i)
                    # print('saved to', filename)

                    # Save camera
                    filename = filestr.format(dir='metadata', category=sequence_category, name=sequence_name, ext='pth')
                    metadata = self.get_metadata(batch, i)
                    torch.save(metadata, filename)

                    # Save evolutions
                    if cfg.run.sample_save_evolutions:
                        torch.save(all_outputs[i], filestr.format(dir='evolutions', category=sequence_category,
                                                                  name=sequence_name, ext='pth'))

        print('Saved samples to: ')
        print(output_dir.absolute())

    def save_pclouds(self, batch, filestr, i, output, sequence_category, sequence_name, save_gt=True):
        from pytorch3d.io import IO
        io = IO()
        if save_gt:
            io.save_pointcloud(data=self.get_gt_pclouds(batch, i), path=filestr.format(dir='gt',
                                                                                       category=sequence_category,
                                                                                       name=sequence_name,
                                                                                       ext='ply'))
        
        # Save generation with colors
        pc: Pointclouds = output[i]
        pred_path = filestr.format(dir='pred', category=sequence_category, name=sequence_name, ext='ply')
        
        # Ensure colors are saved correctly
        if pc.features_packed() is not None:
            # Save with explicit color handling using trimesh
            import trimesh
            points = pc.points_packed().cpu().numpy()
            colors = pc.features_packed().cpu().numpy()
            # Convert from [0, 1] to [0, 255] for trimesh
            colors_255 = (colors * 255).astype(np.uint8)
            pc_trimesh = trimesh.PointCloud(vertices=points, colors=colors_255)
            pc_trimesh.export(pred_path)
        else:
            # Fallback to pytorch3d IO if no colors
            io.save_pointcloud(data=pc, path=pred_path)

        # save binary segmentation if presented
        if pc.features_packed() is not None:
            # with segmentation color, save segmentation results
            assert len(pc.features_list()) == 1
            vc = pc.features_packed()  # (P, 3), human is light blue [0.1, 1.0, 1.0], object light green [0.5, 1.0, 0]
            points = pc.points_packed()  # (P, 3)
            mask_hum = vc[:, 2] > 0.5
            pc_hum, pc_obj = points[mask_hum], points[~mask_hum]
            assert len(pc_hum) > 10, f"Only {len(pc_hum)} human points found in {batch['image_path'][i]}!"
            assert len(pc_obj) > 10, f"Only {len(pc_obj)} object points found in {batch['image_path'][i]}!"
            transl_hum = torch.mean(pc_hum, 0)
            transl_obj = torch.mean(pc_obj, 0)
            scale_hum = torch.sqrt(torch.max(torch.sum((pc_hum - transl_hum) ** 2, -1))).cpu().numpy()
            scale_obj = torch.sqrt(torch.max(torch.sum((pc_obj - transl_obj) ** 2, -1))).cpu().numpy()
            out = {
                "pred_trans": torch.cat([transl_hum, transl_obj], 0).cpu().numpy(),
                "pred_scale": np.array([scale_hum, scale_obj])
            }
            outfile = filestr.format(dir='pred', category=sequence_category, name=sequence_name, ext='pkl')
            # print(f"{torch.sum(mask_hum)}/{len(points)} human points")
            pkl.dump(out, open(outfile, 'wb'))

            # save gt
            pc_gt = self.get_gt_pclouds(batch, i)
            points = pc_gt.points_packed()
            L = len(points)
            pc_hum, pc_obj = points[:L // 2], points[L // 2:]
            transl_hum = torch.mean(pc_hum, 0)
            transl_obj = torch.mean(pc_obj, 0)
            scale_hum = torch.sqrt(torch.max(torch.sum((pc_hum - transl_hum) ** 2, -1))).cpu().numpy()
            scale_obj = torch.sqrt(torch.max(torch.sum((pc_obj - transl_obj) ** 2, -1))).cpu().numpy()
            out = {
                "gt_trans": torch.cat([transl_hum, transl_obj], 0).cpu().numpy(),
                "gt_scale": np.array([scale_hum, scale_obj]),
                "num_smpl": L // 2,
                "samples": points.cpu().numpy()
            }
            outfile = filestr.format(dir='gt', category=sequence_category, name=sequence_name, ext='pkl')
            # print(f"{torch.sum(mask_hum)}/{len(points)} human points")
            pkl.dump(out, open(outfile, 'wb'))

    def get_metadata(self, batch, i):
        metadata = dict(index=i, sequence_name=batch.sequence_name,
                        sequence_category=batch.sequence_category,
                        frame_timestamp=batch.frame_timestamp, camera=batch.camera,
                        image_size_hw=batch.image_size_hw,
                        image_path=batch.image_path, depth_path=batch.depth_path, mask_path=batch.mask_path,
                        bbox_xywh=batch.bbox_xywh, crop_bbox_xywh=batch.crop_bbox_xywh,
                        sequence_point_cloud_path=batch.sequence_point_cloud_path, meta=batch.meta)
        return metadata

    def save_input_image(self, batch, filename, i):
        TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)

    def get_input_image(self, batch, i):
        return batch.image_rgb[i]

    def get_seq_name(self, batch, i):
        sequence_name = batch.sequence_name[i]
        return sequence_name

    def get_seq_category(self, batch, ind=0):
        sequence_category = batch.sequence_category[ind]
        return sequence_category

    def get_batch_size(self, batch):
        return len(batch.image_rgb)

    @torch.no_grad()
    def visualize(
            self,
            cfg: ProjectConfig,
            model: torch.nn.Module,
            dataloader_vis: Iterable,
            accelerator: Accelerator,
            identifier: str = '',
            num_batches: Optional[int] = None,
            output_dir: str = 'vis',
    ):
        from pytorch3d.vis.plotly_vis import plot_scene
        from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
        from pytorch3d.structures import Pointclouds

        # Eval mode
        model.eval()
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        progress_bar: Iterable[FrameData] = metric_logger.log_every(dataloader_vis, cfg.run.print_step_freq, "Vis")

        # Output dir
        output_dir: Path = Path(output_dir)
        (output_dir / 'raw').mkdir(exist_ok=True, parents=True)
        (output_dir / 'pointclouds').mkdir(exist_ok=True, parents=True)
        (output_dir / 'images').mkdir(exist_ok=True, parents=True)
        (output_dir / 'videos').mkdir(exist_ok=True, parents=True)
        (output_dir / 'evolutions').mkdir(exist_ok=True, parents=True)
        (output_dir / 'metadata').mkdir(exist_ok=True, parents=True)

        # Visualize
        wandb_log_dict = {}
        fscores, chamfs = [], []
        for batch_idx, batch in enumerate(progress_bar):
            if num_batches is not None and batch_idx >= num_batches:
                break

            # Sample
            output, all_outputs = model(batch, mode='sample', return_sample_every_n_steps=100,
                                        num_inference_steps=cfg.run.num_inference_steps,
                                        disable_tqdm=(not accelerator.is_main_process))
            output: Pointclouds
            all_outputs: List[Pointclouds]  # list of B Pointclouds, each with a batch size of return_sample_every_n_steps

            # Filenames
            filestr = str(
                output_dir / '{dir}' / f'p-{accelerator.process_index}-b-{batch_idx}-s-{{i:02d}}-{{name}}-{identifier}.{{ext}}')
            filestr_wandb = f'{{dir}}/b-{batch_idx}-{{name}}-s-{{i:02d}}-{{name}}' # identifier=init

            # Not saving raw samples are they are too big
            filename = filestr.format(dir='raw', name='raw', i=0, ext='pth')
            # torch.save({'output': output, 'all_outputs': all_outputs, 'batch': batch}, filename)

            # Save metadata
            metadata = diffusion_utils.get_metadata(batch)
            filename = filestr.format(dir='metadata', name='metadata', i=0, ext='txt')
            Path(filename).write_text(metadata)

            # Save individual samples
            for i in range(len(output)):
                camera, gt_pointcloud = self.preprocess_gt(batch, i) # this should be updated for different datasets

                pred_pointcloud = output[i]
                pred_all_pointclouds = all_outputs[i]

                # Plot using plotly and pytorch3d
                fig = plot_scene({
                    'Pred': {'pointcloud': pred_pointcloud},
                    'GT': {'pointcloud': gt_pointcloud},
                }, ncols=2, viewpoint_cameras=camera, pointcloud_max_points=16_384)

                # Save plot, don't save html, it is too large
                # filename = filestr.format(dir='pointclouds', name='pointclouds', i=i, ext='html')
                # fig.write_html(filename)

                # Add to W&B, don't save html, this is too large
                # filename_wandb = filestr_wandb.format(dir='pointclouds', name='pointclouds', i=i)
                # wandb_log_dict[filename_wandb] = wandb.Html(open(filename), inject=False)

                # Save input images
                filename = filestr.format(dir='images', name='image_rgb', i=i, ext='png')
                TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)

                # Add to W&B
                filename_wandb = filestr_wandb.format(dir='images', name='image_rgb', i=i)
                wandb_log_dict[filename_wandb] = wandb.Image(filename)

                # TODO: compute evaluation error here
                fscore, cd = self.compute_errors(gt_pointcloud, pred_pointcloud)
                fscores.append(fscore)
                chamfs.append(cd)

                # Loop
                for name, pointcloud in (('gt', gt_pointcloud), ('pred', pred_pointcloud)):
                    # Render gt/pred point cloud from given view
                    # these images are saved to vis/images/
                    filename_image = filestr.format(dir='images', name=name, i=i, ext='png')
                    filename_image_wandb = filestr_wandb.format(dir='images', name=name, i=i)
                    diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud,
                                                                         output_file_image=filename_image,
                                                                         cameras=camera,
                                                                         scale_factor=cfg.model.scale_factor)
                    wandb_log_dict[filename_image_wandb] = wandb.Image(filename_image)

                    # Render gt/pred point cloud from rotating view
                    filename_video = filestr.format(dir='videos', name=name, i=i, ext='mp4')
                    filename_video_wandb = filestr_wandb.format(dir='videos', name=name, i=i)
                    diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud,
                                                                         output_file_video=filename_video,
                                                                         num_frames=30,
                                                                         scale_factor=cfg.model.scale_factor)
                    wandb_log_dict[filename_video_wandb] = wandb.Video(filename_video, format="mp4")

                # Render point cloud diffusion evolution
                filename_evo = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
                filename_evo_wandb = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
                diffusion_utils.visualize_pointcloud_evolution_pytorch3d(
                    pointclouds=pred_all_pointclouds, output_file_video=filename_evo, camera=camera)
                wandb_log_dict[filename_evo_wandb] = wandb.Video(filename_evo, format="mp4")

                # Combined comparison panel: Input RGB | GT render | Pred render
                try:
                    from PIL import Image as PILImage
                    input_img = TVF.to_pil_image(self.get_input_image(batch, i))
                    gt_render = diffusion_utils.render_pointcloud_batch_pytorch3d(
                        camera, gt_pointcloud, image_size=224,
                    )  # (1, H, W, 4)
                    pred_render = diffusion_utils.render_pointcloud_batch_pytorch3d(
                        camera, pred_pointcloud, image_size=224,
                    )
                    gt_pil = TVF.to_pil_image(gt_render[0].permute(2, 0, 1)[:3].cpu().clamp(0, 1))
                    pred_pil = TVF.to_pil_image(pred_render[0].permute(2, 0, 1)[:3].cpu().clamp(0, 1))
                    # Resize input to match render size
                    input_resized = input_img.resize(gt_pil.size)
                    panel = PILImage.new('RGB', (gt_pil.width * 3, gt_pil.height))
                    panel.paste(input_resized, (0, 0))
                    panel.paste(gt_pil, (gt_pil.width, 0))
                    panel.paste(pred_pil, (gt_pil.width * 2, 0))
                    panel_key = f'render/comparison_b{batch_idx}_s{i}'
                    wandb_log_dict[panel_key] = wandb.Image(
                        panel, caption=f'Input | GT | Pred (F={fscore:.3f}, CD={cd:.5f})')
                except Exception as e:
                    print(f'[Vis] Could not create comparison panel: {e}')

        # Save to W&B
        if cfg.logging.wandb and accelerator.is_local_main_process:
            # Also log aggregate metrics from visualization
            if fscores:
                wandb_log_dict['vis/fscore_mean'] = np.mean(fscores)
                wandb_log_dict['vis/chamfer_mean'] = np.mean(chamfs)
            wandb.log(wandb_log_dict, commit=False)

        print('Saved visualizations to: ')
        print(output_dir.absolute())
        return fscores, chamfs

    def preprocess_gt(self, batch, i):
        "preprocess for sampling"
        camera = self.get_camera(batch, i)
        gt_pointcloud = self.get_gt_pclouds(batch, i)
        return camera, gt_pointcloud

    def get_camera(self, batch, i):
        return batch.camera[i]

    def get_gt_pclouds(self, batch, i):
        gt_pointcloud = batch.sequence_point_cloud[i]
        return gt_pointcloud

    def compute_errors(self, gt:Pointclouds, pred:Pointclouds, thres:float=0.01):
        """
        compute F-score, CD between gt and prediction
        :param gt:
        :param pred:
        :return:
        """
        assert len(gt.points_list()) == 1, f'found gt points of batch size {len(gt.points_list())}'
        assert len(pred.points_list()) == 1, f'found predicted points of batch size {len(pred.points_list())}'
        gt = gt.points_packed().cpu().numpy()
        pred = pred.points_packed().cpu().numpy()

        if np.any(np.isnan(gt)) or np.any(np.isnan(pred)):
            print("Warning: found NaN values in predicted points!")
            return 0, 100.

        chamf, fscore = self.compute_fscore_chamf(gt, pred, thres)
        return fscore, chamf

    def compute_fscore_chamf(self, gt, pred, thres):
        """
        compute fscore with numpy array, gt and pred are both (N, 3)
        """
        chamf, d1, d2 = chamfer_distance(gt, pred, ret_intermediate=True)
        recall = float(sum(d < thres for d in d2)) / float(len(d2))
        precision = float(sum(d < thres for d in d1)) / float(len(d1))
        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
        return chamf, fscore


###############################################################################
# Phase 2: Dual-Branch Flow Matching Trainer
###############################################################################
class TrainerFlowMatching(Trainer):
    """
    Trainer for Phase 2 dual-branch flow matching.
    Overrides __init__ to avoid importing model/ package (pvcnn CUDA JIT).
    Overrides train() with a flow-matching-specific training loop.
    """

    def __init__(self, cfg: ProjectConfig):
        # --- Accelerator & logging (same as base) ---
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu,
            gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )
        training_utils.setup_distributed_print(accelerator.is_main_process)

        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.init(
                project=cfg.logging.wandb_project, name=cfg.run.name,
                job_type=cfg.run.job, config=OmegaConf.to_container(cfg),
            )

        print(f'Current working directory: {os.getcwd()}')
        training_utils.set_seed(cfg.run.seed)

        # --- Model: PVCNN-based flow matching (fast, baseline architecture) ---
        # Add baseline to path for PVCNN CUDA ops
        import sys as _sys
        _baseline_dir = os.path.join(cfg.run.code_dir_abs, 'baseline')
        if _baseline_dir not in _sys.path:
            _sys.path.insert(0, _baseline_dir)

        from importlib.util import spec_from_file_location, module_from_spec
        # Load FlowMatchingTrainer / euler_ode_sample from original module
        _spec = spec_from_file_location(
            'dual_branch_flow_matching',
            os.path.join(cfg.run.code_dir_abs, 'model', 'dual_branch_flow_matching.py'),
        )
        _mod = module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        # Load PVCNN-based model
        _pvcnn_spec = spec_from_file_location(
            'pvcnn_flow_matching',
            os.path.join(cfg.run.code_dir_abs, 'model', 'pvcnn_flow_matching.py'),
        )
        _pvcnn_mod = module_from_spec(_pvcnn_spec)
        _pvcnn_spec.loader.exec_module(_pvcnn_mod)

        model = _pvcnn_mod.PVCNNFlowMatchingModel(
            video_channels=cfg.model.video_channels,
            video_input_channels=getattr(cfg.model, 'video_input_channels', cfg.model.video_channels),
            point_channels=cfg.model.point_channels,
            mask_channels=cfg.model.mask_channels,
        )
        self._fm_trainer = _mod.FlowMatchingTrainer(
            model, lambda_video=cfg.model.lambda_video, lambda_3d=cfg.model.lambda_3d,
        )
        self._euler_ode_sample = _mod.euler_ode_sample

        num_params = sum(p.numel() for p in model.parameters())
        print(f'[FlowMatching] Model: {num_params / 1e6:.2f}M parameters')

        self.model_ema = None

        # --- Optimizer & scheduler (lr scales with total batch like baseline) ---
        optimizer = training_utils.get_optimizer(cfg, model, accelerator)
        # Apply linear lr scaling: effective_lr = base_lr * num_gpus * batch_size_per_gpu
        if cfg.optimizer.scale_learning_rate_with_batch_size:
            scaled_lr = cfg.optimizer.lr * accelerator.num_processes * cfg.dataloader.batch_size
            for pg in optimizer.param_groups:
                pg['lr'] = scaled_lr
            print(f'[FM] Scaled lr: {accelerator.num_processes} GPUs * bs {cfg.dataloader.batch_size} * {cfg.optimizer.lr} = {scaled_lr}')
        scheduler = training_utils.get_scheduler(cfg, optimizer)

        # --- Resume ---
        self.train_state = self._load_fm_checkpoint(cfg, model, optimizer, scheduler)

        # --- Dataset ---
        mc = cfg.model
        if mc.use_real_data:
            from dataset.fm_dataset import FlowMatchingDataset
            from torch.utils.data import DataLoader
            import pickle as _pkl

            preprocessed_dir = mc.preprocessed_dir
            if not os.path.isabs(preprocessed_dir):
                preprocessed_dir = os.path.join(cfg.run.code_dir_abs, preprocessed_dir)

            # Discover sequences from preprocessed dir
            all_entries = sorted(os.listdir(preprocessed_dir))
            # ProciGen sequences are directories (train), BEHAVE are .pt files (test)
            train_seqs = [e for e in all_entries if os.path.isdir(os.path.join(preprocessed_dir, e))]
            # Also load split file to verify if available
            split_file = getattr(cfg.dataset, 'split_file', '')
            if split_file and os.path.isfile(split_file):
                split_data = _pkl.load(open(split_file, 'rb'))
                # Extract ProciGen sequence names from train paths
                split_train_seqs = set()
                for p in split_data['train']:
                    seq = p.split('/')[0]
                    split_train_seqs.add(seq)
                # Filter to only sequences present in both preprocessed and split
                train_seqs = [s for s in train_seqs if s in split_train_seqs]
                print(f'[FlowMatching] Filtered to {len(train_seqs)} train sequences from split file')

            consolidated_dir = mc.consolidated_dir
            if consolidated_dir and not os.path.isabs(consolidated_dir):
                consolidated_dir = os.path.join(cfg.run.code_dir_abs, consolidated_dir)

            cache_file = mc.cache_file
            if cache_file and not os.path.isabs(cache_file):
                cache_file = os.path.join(cfg.run.code_dir_abs, cache_file)

            # cache_in_memory: True if a cache_file is provided (fast load from single file),
            # otherwise False to avoid slow sequential preload of 1M+ files from HDD.
            # Users should run preload_to_ram_cache.py first to create the cache file.
            use_cache = bool(cache_file) and os.path.isfile(cache_file)
            dataset_train = FlowMatchingDataset(
                preprocessed_dir=preprocessed_dir,
                sequence_dirs=train_seqs,
                num_frames=mc.num_frames,
                video_h=mc.video_h,
                video_w=mc.video_w,
                num_points=mc.num_points,
                point_channels=mc.point_channels,
                split='train',
                cache_in_memory=use_cache,
                consolidated_dir=consolidated_dir,
                cache_file=cache_file,

            )
            num_wk = cfg.dataloader.num_workers
            loader_kwargs = dict(
                batch_size=cfg.dataloader.batch_size,
                shuffle=True,
                num_workers=num_wk,
                pin_memory=True,
                drop_last=True,
            )
            if num_wk > 0:
                loader_kwargs['persistent_workers'] = True
                loader_kwargs['prefetch_factor'] = 4
            self.dataloader_train = DataLoader(dataset_train, **loader_kwargs)
            print(f'[FlowMatching] Real dataset: {len(dataset_train)} samples, '
                  f'{len(self.dataloader_train)} batches/epoch')
        else:
            self.dataloader_train = None
            print('[FlowMatching] Using synthetic data (use_real_data=False)')

        self.dataloader_val = None
        self.dataloader_vis = None
        self.total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes

        # --- Accelerator prepare ---
        if self.dataloader_train is not None:
            model, optimizer, scheduler, self.dataloader_train = accelerator.prepare(
                model, optimizer, scheduler, self.dataloader_train)
        else:
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        # Update _fm_trainer to use the DDP-wrapped model for proper gradient sync
        self._fm_trainer.model = model

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.accelerator = accelerator
        self.loss_sep = None

    def _load_fm_checkpoint(self, cfg, model, optimizer, scheduler):
        """Auto-resume: detect checkpoint-latest.pth and restore full training state.
        Set checkpoint.resume=none to force training from scratch."""
        # Explicit skip
        if cfg.checkpoint.resume and cfg.checkpoint.resume.lower() == 'none':
            print('[FM] checkpoint.resume=none, starting from scratch')
            return training_utils.TrainState()

        # Auto-detect checkpoint path
        ckpt_path = None
        if cfg.checkpoint.resume and cfg.checkpoint.resume not in ('', 'test'):
            ckpt_path = cfg.checkpoint.resume

        if ckpt_path is None or not os.path.isfile(ckpt_path):
            # Try auto-detect from output dir
            auto_path = os.path.join(
                cfg.run.code_dir_abs, f'outputs/{cfg.run.name}/single/checkpoint-latest.pth'
            )
            if os.path.isfile(auto_path):
                ckpt_path = auto_path
                print(f'[FM] Auto-detected checkpoint: {ckpt_path}')

        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print('[FM] No checkpoint found, starting from scratch')
            return training_utils.TrainState()

        print(f'[FM] Loading checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Restore model weights
        state_dict = ckpt.get('model', ckpt)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[FM]  Missing keys: {len(missing)}')
        if unexpected:
            print(f'[FM]  Unexpected keys: {len(unexpected)}')

        # Restore optimizer
        if cfg.checkpoint.resume_training_optimizer and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print('[FM] Restored optimizer state')

        # Restore scheduler
        if cfg.checkpoint.resume_training_scheduler and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            print('[FM] Restored scheduler state')

        # Restore training state
        step = ckpt.get('step', 0)
        epoch = ckpt.get('epoch', 0)
        best_val = ckpt.get('best_val', None)
        print(f'[FM] Resumed: step={step}, epoch={epoch}')
        return training_utils.TrainState(epoch=epoch, step=step, best_val=best_val)

    def _generate_synthetic_batch(self, device):
        """Synthetic data batch (placeholder for real data loader)."""
        B = self.cfg.dataloader.batch_size
        T = self.cfg.model.num_frames
        # BUG FIX: Must use video_input_channels (4: RGB+mask) not video_channels (3: RGB only)
        # The model input is 4-channel, the output is 3-channel (RGB only).
        C_v_in = getattr(self.cfg.model, 'video_input_channels', self.cfg.model.video_channels)
        H, W = self.cfg.model.video_h, self.cfg.model.video_w
        C_m = self.cfg.model.mask_channels
        N = self.cfg.model.num_points
        C_3d = self.cfg.model.point_channels

        x_1_video = torch.randn(B, T, C_v_in, H, W, device=device) * 0.5
        x_1_3d = torch.randn(B, T, N, C_3d, device=device) * 0.5
        mask_feat = torch.randn(B, T, C_m, H, W, device=device)
        return x_1_video, x_1_3d, mask_feat

    def train(self, cfg: ProjectConfig):
        """Flow matching training loop with real or synthetic data."""
        train_state = self.train_state
        model, optimizer, scheduler = self.model, self.optimizer, self.scheduler
        accelerator = self.accelerator
        device = accelerator.device
        use_real = self.dataloader_train is not None

        total_bs = cfg.dataloader.batch_size * accelerator.num_processes
        print(f'[FM] Start training | bs={cfg.dataloader.batch_size}x{accelerator.num_processes}gpu={total_bs} | '
              f'steps={cfg.run.max_steps} | {"real" if use_real else "synth"} | '
              f'resume step={train_state.step} epoch={train_state.epoch}')

        model.train()
        _train_start_time = time.time()
        _start_step = train_state.step

        while train_state.step < cfg.run.max_steps:
            # --- Epoch-based iteration for real data ---
            if use_real:
                data_iter = iter(self.dataloader_train)
            else:
                data_iter = None

            batch_in_epoch = 0
            while train_state.step < cfg.run.max_steps:
                # Get batch
                if use_real:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break  # end of epoch
                    x_1_video = batch['x_video_human'].to(device)
                    x_1_3d = batch['x_3d'].to(device)
                    mask_feat = batch['mask_features'].to(device)
                else:
                    x_1_video, x_1_3d, mask_feat = self._generate_synthetic_batch(device)

                x_0_video = torch.randn_like(x_1_video)
                x_0_3d = torch.randn_like(x_1_3d)

                with accelerator.accumulate(model):
                    loss, log = self._fm_trainer.compute_loss(
                        x_0_video, x_1_video, x_0_3d, x_1_3d, mask_features=mask_feat,
                    )
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if cfg.optimizer.clip_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        scheduler.step()
                        train_state.step += 1

                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        print(f'Loss is {loss_value}, stopping training')
                        sys.exit(90)

                if accelerator.sync_gradients:
                    # Compact logging
                    if train_state.step % cfg.run.print_step_freq == 0:
                        lr_now = optimizer.param_groups[0]['lr']
                        elapsed = time.time() - _train_start_time
                        steps_done = train_state.step - _start_step
                        if steps_done > 0:
                            eta_s = elapsed / steps_done * (cfg.run.max_steps - train_state.step)
                            # Format ETA as compact string
                            eta_h = eta_s / 3600
                            if eta_h >= 24:
                                eta_str = f'{eta_h/24:.1f}d'
                            else:
                                eta_str = f'{eta_h:.1f}h'
                        else:
                            eta_str = '?'
                        pct = 100 * train_state.step / cfg.run.max_steps
                        print(f'E{train_state.epoch} S{train_state.step}/{cfg.run.max_steps}({pct:.1f}%) '
                              f'L={log["loss_total"]:.4f}(v={log["loss_video"]:.4f},3d={log["loss_3d"]:.4f}) '
                              f'lr={lr_now:.1e} ETA={eta_str}')

                    if cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0:
                        fm_log_dict = {
                            'train_loss': log['loss_total'],
                            'train_loss_video': log['loss_video'],
                            'train_loss_3d': log['loss_3d'],
                            'lr': optimizer.param_groups[0]['lr'],
                            't_mean': log['t_mean'],
                            'step': train_state.step,
                            'epoch': train_state.epoch,
                        }

                        # Periodically sample and log reconstructed images
                        if cfg.run.vis_freq > 0 and train_state.step % cfg.run.vis_freq == 0:
                            model.eval()
                            with torch.no_grad():
                                vis_x0_video = torch.randn_like(x_1_video)
                                # Preserve the mask channel from the real data
                                # so the model sees correct mask conditioning
                                C_rgb = cfg.model.video_channels  # 3
                                if vis_x0_video.shape[2] > C_rgb:
                                    vis_x0_video[:, :, C_rgb:] = x_1_video[:, :, C_rgb:]
                                vis_x0_3d = torch.randn_like(x_1_3d)
                                raw_model = accelerator.unwrap_model(model)
                                gen_video, gen_3d = self._euler_ode_sample(
                                    raw_model, vis_x0_video, vis_x0_3d,
                                    num_steps=20, mask_features=mask_feat,
                                )
                                gt_frame = x_1_video[0, 0, :3].clamp(0, 1)
                                gen_frame = gen_video[0, 0, :3].clamp(0, 1)
                                comparison = torch.cat([gt_frame, gen_frame], dim=-1)
                                fm_log_dict['render/video_gt_vs_gen'] = wandb.Image(
                                    TVF.to_pil_image(comparison.cpu()),
                                    caption=f'Step {train_state.step} | Left: GT, Right: Generated')
                                pts_gt = x_1_3d[0, 0, :, :3].cpu()
                                pts_gen = gen_3d[0, 0, :, :3].cpu()
                                try:
                                    import matplotlib
                                    matplotlib.use('Agg')
                                    import matplotlib.pyplot as plt
                                    fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': '3d'})
                                    for ax, pts, title in [(axes[0], pts_gt, 'GT 3D'), (axes[1], pts_gen, 'Gen 3D')]:
                                        ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), pts[:, 2].numpy(), s=1, alpha=0.6)
                                        ax.set_title(title)
                                    fig.tight_layout()
                                    fm_log_dict['render/3d_gt_vs_gen'] = wandb.Image(fig,
                                        caption=f'Step {train_state.step} | 3D point clouds')
                                    plt.close(fig)
                                except Exception as e:
                                    print(f'[Vis] 3D scatter failed: {e}')
                            model.train()

                        wandb.log(fm_log_dict, step=train_state.step)

                    # Checkpoint
                    if accelerator.is_main_process and train_state.step % cfg.run.checkpoint_freq == 0:
                        self.save_checkpoint(
                            accelerator, cfg, model, self.model_ema, optimizer, scheduler, train_state,
                        )

                batch_in_epoch += 1

            # End of epoch
            if use_real:
                train_state.epoch += 1
                print(f'Epoch {train_state.epoch - 1} done ({batch_in_epoch} batches)')
            else:
                break

        print(f'Training done at {datetime.datetime.now()} | final: {train_state}')
        self.cleanup_gpu_memory()
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
            time.sleep(5)


###############################################################################
# Phase 3: Joint 3DGS Optimization Trainer
###############################################################################
class TrainerJoint3DGS(Trainer):
    """
    Trainer for Step 4: Multi-Region Contact-Aware Joint 3DGS Optimization.

    Overrides __init__ to build GaussianModel + SE(3) transforms + loss modules.
    Overrides train() with the full Step 4 optimization loop including:
      - SE(3) coordinate registration
      - Multi-region rendering loss
      - Contact / 2D projection / penetration / temporal smoothness losses
    """

    def __init__(self, cfg: ProjectConfig):
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision='no',  # 3DGS needs full precision
            cpu=cfg.run.cpu,
            kwargs_handlers=[ddp_kwargs],
        )
        training_utils.setup_distributed_print(accelerator.is_main_process)

        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.init(
                project=cfg.logging.wandb_project, name=cfg.run.name,
                job_type=cfg.run.job, config=OmegaConf.to_container(cfg),
            )

        print(f'Current working directory: {os.getcwd()}')
        training_utils.set_seed(cfg.run.seed)

        # --- Import Step 4 components ---
        import glob as _glob
        sys.path.insert(0, cfg.run.code_dir_abs)
        from scripts.step4_joint_optimization import (
            SE3Transform, JointRenderer, VolumetricSMPLSDF,
            step4_training_step, load_step1_outputs, load_gs_init,
        )
        from scripts.joint_3dgs_optimization import (
            GaussianModel, SimpleProjectionRenderer, load_image, load_mask,
        )
        self._step4_training_step = step4_training_step

        mc = cfg.model  # Joint3DGSModelConfig
        H, W = mc.image_height, mc.image_width
        device = accelerator.device

        code_dir = cfg.run.code_dir_abs

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.join(code_dir, p)

        # --- Load video frames ---
        pt_path = _resolve(mc.preprocessed_pt) if mc.preprocessed_pt else ''
        if pt_path and os.path.isfile(pt_path):
            print(f'[Step4] Loading preprocessed data: {pt_path}')
            cached = torch.load(pt_path, map_location='cpu', weights_only=False)
            import torch.nn.functional as _F
            def _maybe_resize(t, h, w, mode='bilinear'):
                if t.shape[-2] != h or t.shape[-1] != w:
                    return _F.interpolate(t, size=(h, w), mode=mode,
                                          align_corners=False if mode == 'bilinear' else None)
                return t
            self._frames = [_maybe_resize(f.unsqueeze(0), H, W).squeeze(0).to(device)
                            for f in cached['frames']]
            self._human_masks = [_maybe_resize(m.unsqueeze(0), H, W, mode='nearest').squeeze(0).squeeze(0).to(device)
                                 for m in cached.get('masks_human', [])]
            self._object_masks = [_maybe_resize(m.unsqueeze(0), H, W, mode='nearest').squeeze(0).squeeze(0).to(device)
                                  for m in cached.get('masks_object', [])]
        else:
            frames_dir = _resolve(mc.frames_dir)
            frame_paths = sorted(_glob.glob(os.path.join(frames_dir, '*.png')))
            if not frame_paths:
                frame_paths = sorted(_glob.glob(os.path.join(frames_dir, '*.jpg')))
            assert len(frame_paths) > 0, f'No frames found in {frames_dir}'
            self._frames = [load_image(p, H, W).to(device) for p in frame_paths]
            masks_human_dir = _resolve(mc.masks_human_dir) if mc.masks_human_dir else ''
            masks_object_dir = _resolve(mc.masks_object_dir) if mc.masks_object_dir else ''
            hmask_paths = sorted(_glob.glob(os.path.join(masks_human_dir, '*.png'))) if masks_human_dir else []
            omask_paths = sorted(_glob.glob(os.path.join(masks_object_dir, '*.png'))) if masks_object_dir else []
            self._human_masks = [load_mask(p, H, W).to(device) for p in hmask_paths] if hmask_paths else []
            self._object_masks = [load_mask(p, H, W).to(device) for p in omask_paths] if omask_paths else []

        num_frames = len(self._frames)
        print(f'[Step4] {num_frames} frames loaded')

        # --- Load Step 1 outputs (soft masks, SMPL-H, keypoints) ---
        processed_dir = _resolve(mc.processed_dir) if mc.processed_dir else ''
        if processed_dir and os.path.isdir(processed_dir):
            self._step1_data = load_step1_outputs(processed_dir, device)
        else:
            self._step1_data = {
                'masks_visible': [], 'masks_primary_occ': [],
                'masks_secondary_occ': [], 'smplh_params': [],
                'keypoints_2d': [], 'kp_confidence': [], 'camera_K': None,
            }

        # Fallback masks
        if not self._step1_data['masks_visible']:
            self._step1_data['masks_visible'] = (
                self._object_masks if self._object_masks
                else [torch.ones(H, W, device=device)] * num_frames
            )
        if not self._step1_data['masks_primary_occ']:
            self._step1_data['masks_primary_occ'] = (
                self._human_masks if self._human_masks
                else [torch.zeros(H, W, device=device)] * num_frames
            )
        if not self._step1_data['masks_secondary_occ']:
            self._step1_data['masks_secondary_occ'] = [
                torch.zeros(H, W, device=device)
            ] * num_frames

        # --- Gaussian models (init from Phase 2/3 or random) ---
        phase2_path = getattr(mc, 'phase2_output', '')
        if phase2_path:
            phase2_path = _resolve(phase2_path)
        if phase2_path and os.path.isfile(phase2_path):
            print(f'[Step4] Loading Phase 2 output: {phase2_path}')
            phase2_data = torch.load(phase2_path, map_location='cpu', weights_only=False)
            gen_3d = phase2_data['generated_3d']
            gaussians = gen_3d[0, 0]
            n_total = gaussians.shape[0]
            n_hum = n_total // 2
            self._human_gs = GaussianModel.from_phase2(gaussians[:n_hum]).to(device)
            self._object_gs = GaussianModel.from_phase2(gaussians[n_hum:]).to(device)
        else:
            self._human_gs = GaussianModel(num_points=mc.num_points_human, init_extent=0.5).to(device)
            self._object_gs = GaussianModel(num_points=mc.num_points_object, init_extent=0.3).to(device)

        # --- SE(3) transforms ---
        self._se3_human = SE3Transform(init_translation=(0., 0., 2.)).to(device)
        self._se3_object = SE3Transform(init_translation=(0., 0., 2.)).to(device)

        # --- Renderer ---
        base_renderer = SimpleProjectionRenderer(H, W, focal=mc.focal).to(device)
        self._joint_renderer = JointRenderer(base_renderer, self._se3_human, self._se3_object).to(device)

        # --- SDF module ---
        self._sdf_module = (
            VolumetricSMPLSDF(resolution=mc.sdf_resolution, padding=mc.sdf_padding).to(device)
            if mc.enable_penetration else None
        )

        # --- Camera intrinsics ---
        self._focal = mc.focal
        self._cx, self._cy = W / 2.0, H / 2.0
        if self._step1_data['camera_K'] is not None:
            K = self._step1_data['camera_K']
            self._focal = K[0, 0].item()
            self._cx = K[0, 2].item()
            self._cy = K[1, 2].item()

        # --- Optimizer (separate LR per param group) ---
        param_groups = [
            {'params': [self._human_gs.xyz, self._object_gs.xyz], 'lr': mc.lr_xyz},
            {'params': [self._human_gs.opacity, self._object_gs.opacity], 'lr': mc.lr_opacity},
            {'params': [self._human_gs.scaling, self._object_gs.scaling], 'lr': mc.lr_scaling},
            {'params': [self._human_gs.rotation, self._object_gs.rotation], 'lr': mc.lr_rotation},
            {'params': [self._human_gs.shs, self._object_gs.shs], 'lr': mc.lr_color},
            {'params': [self._se3_human.translation, self._se3_object.translation],
             'lr': mc.lr_se3_translation},
            {'params': [self._se3_human.axis_angle, self._se3_object.axis_angle],
             'lr': mc.lr_se3_rotation},
        ]
        optimizer = torch.optim.Adam(param_groups)

        # --- LR Scheduler: cosine annealing to 1% of initial LR ---
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=mc.num_iters, eta_min=mc.lr_xyz * 0.01,
        )

        # --- Hand joint indices for contact loss ---
        # SMPL-H: 20=L_wrist, 21=R_wrist, 22-51=hand joints
        self._hand_indices = list(range(20, 52)) if mc.enable_contact else None

        self.model = None
        self.model_ema = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_state = training_utils.TrainState()
        self.dataloader_train = None
        self.dataloader_val = None
        self.dataloader_vis = None
        self.total_batch_size = 1
        self.cfg = cfg
        self.accelerator = accelerator
        self.loss_sep = None

    def train(self, cfg: ProjectConfig):
        """Step 4 optimization loop with all physics-aware losses."""
        mc = cfg.model
        train_state = self.train_state
        optimizer = self.optimizer
        scheduler = self.scheduler
        accelerator = self.accelerator
        num_frames = len(self._frames)

        print(f'***** Starting Step 4 Joint Optimization at {datetime.datetime.now()} *****')
        print(f'    Num frames = {num_frames}')
        print(f'    Num iters = {mc.num_iters}')
        print(f'    Losses: render={mc.w_visible}/{mc.w_primary_occ}/{mc.w_secondary_occ} '
              f'contact={mc.lambda_contact} j2d={mc.lambda_j2d} '
              f'pen={mc.lambda_pen} acc={mc.lambda_acc}')

        from tqdm import tqdm

        pose_history = []
        pbar = tqdm(range(1, mc.num_iters + 1), desc='Step4 Joint Opt')
        _train_start = time.time()

        def _get_se3_pose():
            return torch.cat([
                self._se3_object.axis_angle,
                self._se3_object.translation,
            ]).detach().clone()

        for step in pbar:
            _step_start = time.time()
            idx = (step - 1) % num_frames

            gt_image = self._frames[idx]
            m_vis = self._step1_data['masks_visible'][idx % len(self._step1_data['masks_visible'])]
            m_pri = self._step1_data['masks_primary_occ'][idx % len(self._step1_data['masks_primary_occ'])]
            m_sec = self._step1_data['masks_secondary_occ'][idx % len(self._step1_data['masks_secondary_occ'])]

            # Per-frame SMPL-H data
            smpl_joints = None
            smpl_verts = None
            smpl_faces_t = None
            kp2d = None
            kp_conf = None

            if self._step1_data['smplh_params'] and idx < len(self._step1_data['smplh_params']):
                params = self._step1_data['smplh_params'][idx]
                smpl_joints = params.get('joints_3d')
                smpl_verts = params.get('vertices')
                smpl_faces_t = params.get('faces')
                if smpl_faces_t is not None:
                    smpl_faces_t = smpl_faces_t.long()

            if self._step1_data['keypoints_2d'] and idx < len(self._step1_data['keypoints_2d']):
                kp2d = self._step1_data['keypoints_2d'][idx]
                kp_conf = self._step1_data['kp_confidence'][idx]

            # Temporal: sliding window with gradient flow through current SE(3)
            se3_prev_det = pose_history[-2] if len(pose_history) >= 2 else None
            se3_next_det = pose_history[-1] if len(pose_history) >= 1 else None

            log = self._step4_training_step(
                human_gs=self._human_gs,
                object_gs=self._object_gs,
                joint_renderer=self._joint_renderer,
                gt_image=gt_image,
                mask_visible=m_vis,
                mask_primary_occ=m_pri,
                mask_secondary_occ=m_sec,
                optimizer=optimizer,
                smpl_joints_3d=smpl_joints,
                hand_joint_indices=self._hand_indices,
                keypoints_2d=kp2d,
                kp_confidence=kp_conf,
                smpl_vertices=smpl_verts,
                smpl_faces=smpl_faces_t,
                sdf_module=self._sdf_module,
                se3_pose_prev_detached=se3_prev_det,
                se3_pose_next_detached=se3_next_det,
                se3_object_module=self._se3_object if mc.enable_temporal else None,
                w_visible=mc.w_visible,
                w_primary=mc.w_primary_occ,
                w_secondary=mc.w_secondary_occ,
                lambda_ssim=mc.lambda_ssim,
                lambda_contact=mc.lambda_contact if mc.enable_contact else 0.0,
                lambda_j2d=mc.lambda_j2d if mc.enable_j2d else 0.0,
                lambda_pen=mc.lambda_pen if mc.enable_penetration else 0.0,
                lambda_acc=mc.lambda_acc if mc.enable_temporal else 0.0,
                focal=self._focal,
                cx=self._cx,
                cy=self._cy,
                conf_threshold=mc.j2d_conf_threshold,
            )
            train_state.step = step

            # Step LR scheduler
            if scheduler is not None:
                scheduler.step()

            # Cache pose after optimization step
            pose_history.append(_get_se3_pose())
            if len(pose_history) > 3:
                pose_history.pop(0)

            # Logging with ETA
            if step % 100 == 0:
                elapsed = time.time() - _train_start
                eta_s = elapsed / step * (mc.num_iters - step)
                if eta_s >= 3600:
                    eta_str = f'{eta_s / 3600:.1f}h'
                elif eta_s >= 60:
                    eta_str = f'{eta_s / 60:.1f}m'
                else:
                    eta_str = f'{eta_s:.0f}s'
                lr_now = optimizer.param_groups[0]['lr']
                pbar.set_postfix(
                    render=f"{log['loss_render']:.4f}",
                    contact=f"{log['loss_contact']:.4f}",
                    pen=f"{log['loss_penetration']:.5f}",
                    acc=f"{log['loss_temporal']:.5f}",
                    lr=f"{lr_now:.1e}",
                    eta=eta_str,
                )

            # Wandb logging
            if cfg.logging.wandb and accelerator.is_main_process and step % cfg.run.log_step_freq == 0:
                log_dict = {
                    'train_loss': log['loss_total'],
                    'train_loss_render': log['loss_render'],
                    'train_loss_contact': log['loss_contact'],
                    'train_loss_j2d': log['loss_j2d'],
                    'train_loss_penetration': log['loss_penetration'],
                    'train_loss_temporal': log['loss_temporal'],
                    'lr': optimizer.param_groups[0]['lr'],
                    'step': step,
                }

                if step % cfg.run.vis_freq == 0:
                    with torch.no_grad():
                        rendered, _, _ = self._joint_renderer(self._human_gs, self._object_gs)
                        gt_vis = gt_image.clamp(0, 1)
                        rendered_vis = rendered.clamp(0, 1)
                        comparison = torch.cat([gt_vis, rendered_vis], dim=-1)
                        comparison_pil = TVF.to_pil_image(comparison.cpu())
                        log_dict['render/gt_vs_pred'] = wandb.Image(
                            comparison_pil, caption=f'Step {step} | Left: GT, Right: Rendered')

                wandb.log(log_dict, step=step)

            # Checkpoint
            if accelerator.is_main_process and (step % mc.save_every == 0 or step == mc.num_iters):
                ckpt = {
                    'step': step,
                    'human_gs': self._human_gs.state_dict(),
                    'object_gs': self._object_gs.state_dict(),
                    'se3_human': self._se3_human.state_dict(),
                    'se3_object': self._se3_object.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                }
                ckpt_path = os.path.join(os.getcwd(), f'step4_ckpt_{step:06d}.pt')
                torch.save(ckpt, ckpt_path)
                print(f'\n  [Save] {ckpt_path}')

        total_time = time.time() - _train_start
        print(f'Ending Step 4 optimization at: {datetime.datetime.now()} ({total_time / 60:.1f}min)')
        print(f'Final train state: {train_state}')
        self.cleanup_gpu_memory()
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
            time.sleep(5)
