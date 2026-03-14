"""
Dataset for Phase 2 Flow Matching training on real ProciGen data.

Supports three modes:
  1. Consolidated: loads pre-merged per-sequence .pt files (fast)
  2. Per-frame + preload: reads all per-frame files sequentially at startup,
     preprocesses and caches in RAM (~140GB for full dataset). Converts HDD
     random reads to sequential reads — much faster on spinning disks.
  3. Per-frame lazy: loads individual .pt files on demand (slow on HDD)

Usage:
  # Fastest: run consolidation once, then use consolidated_dir
  python scripts/preprocess_fm_cache.py --src preprocessed/phase1 --dst preprocessed/phase1_consolidated

  # Or just set cache_in_memory=True (default) for automatic preload at startup
"""
import os
import re
import sys
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class _ProgressFileWrapper:
    """Seekable file wrapper that tracks read progress for torch.load."""

    def __init__(self, fobj, total_size, update_interval=0.5):
        self._f = fobj
        self._total = total_size
        self._bytes_read = 0
        self._last_print = 0
        self._t0 = time.time()
        self._interval = update_interval

    def read(self, n=-1):
        data = self._f.read(n)
        self._bytes_read += len(data)
        self._maybe_print()
        return data

    def readinto(self, b):
        n = self._f.readinto(b)
        if n:
            self._bytes_read += n
            self._maybe_print()
        return n

    def readline(self):
        data = self._f.readline()
        self._bytes_read += len(data)
        self._maybe_print()
        return data

    def seek(self, offset, whence=0):
        result = self._f.seek(offset, whence)
        self._bytes_read = self._f.tell()
        return result

    def tell(self):
        return self._f.tell()

    def _maybe_print(self):
        now = time.time()
        if now - self._last_print < self._interval:
            return
        self._last_print = now
        pct = 100.0 * self._bytes_read / self._total if self._total > 0 else 0
        elapsed = now - self._t0
        speed_mb = self._bytes_read / (1024**2) / elapsed if elapsed > 0 else 0
        if self._bytes_read > 0 and elapsed > 1:
            eta_s = (self._total - self._bytes_read) / (self._bytes_read / elapsed)
            eta_str = f'{eta_s:.0f}s'
        else:
            eta_str = '?'
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f'\r  [Cache] |{bar}| {pct:5.1f}% {speed_mb:.0f}MB/s ETA={eta_str}', end='', flush=True)


def _load_with_progress(cache_file, file_size):
    """Load a torch cache file with a real progress bar."""
    with open(cache_file, 'rb') as f:
        wrapper = _ProgressFileWrapper(f, file_size)
        data = torch.load(wrapper, map_location='cpu', weights_only=False)
    return data


class FlowMatchingDataset(Dataset):
    """Fast dataset loading with sequential preload for HDD-friendly access."""

    def __init__(
        self,
        preprocessed_dir: str,
        sequence_dirs: list,
        num_frames: int = 4,
        video_h: int = 32,
        video_w: int = 32,
        num_points: int = 256,
        point_channels: int = 14,
        split: str = 'train',
        cache_in_memory: bool = True,
        consolidated_dir: str = '',
        cache_file: str = '',
    ):
        super().__init__()
        self.num_frames = num_frames
        self.video_h = video_h
        self.video_w = video_w
        self.num_points = num_points
        self.point_channels = point_channels
        self.split = split
        self.cache_in_memory = cache_in_memory
        self._cache = {}

        # Mode 1: Load from pre-built cache file (fastest startup)
        if cache_file and os.path.isfile(cache_file):
            file_size = os.path.getsize(cache_file)
            file_size_gb = file_size / (1024**3)
            print(f'[FMDataset] Loading cache {cache_file} ({file_size_gb:.1f}GB)...')
            t0 = time.time()
            self._cache = _load_with_progress(cache_file, file_size)
            elapsed = time.time() - t0
            print(f'\n[FMDataset] Loaded {len(self._cache)} entries in {elapsed:.0f}s '
                  f'({self._estimate_cache_memory_gb():.1f}GB)')
            self.cache_in_memory = True
            self.use_consolidated = False
            self._init_per_frame(preprocessed_dir, sequence_dirs, num_frames)
            return

        # Mode 2: Consolidated per-sequence files
        self.consolidated_dir = consolidated_dir
        self.use_consolidated = bool(consolidated_dir) and os.path.isdir(consolidated_dir)

        if self.use_consolidated:
            self._init_consolidated(consolidated_dir, sequence_dirs, num_frames)
        else:
            # Mode 3: Per-frame files with optional preload
            self._init_per_frame(preprocessed_dir, sequence_dirs, num_frames)
            if self.cache_in_memory:
                self._preload_all_to_memory()

    def _init_consolidated(self, consolidated_dir, sequence_dirs, num_frames):
        """Index consolidated .pt files (one per sequence)."""
        print(f"[FMDataset] Loading consolidated data from {consolidated_dir}...")
        self.sequences = []
        self._seq_data = {}

        for seq_name in sequence_dirs:
            pt_path = os.path.join(consolidated_dir, f'{seq_name}.pt')
            if not os.path.isfile(pt_path):
                continue
            self._seq_data[seq_name] = pt_path
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            n_frames = len(data['sorted_frame_ids'])
            if n_frames < num_frames:
                continue
            self.sequences.append((seq_name, data['sorted_frame_ids'], n_frames))
            if self.cache_in_memory:
                self._cache[seq_name] = data

        stride = max(1, num_frames // 2)
        self.samples = []
        for seq_idx, (seq_name, frame_ids, n) in enumerate(self.sequences):
            for start in range(0, n - num_frames + 1, stride):
                self.samples.append((seq_idx, start))

        print(f"[FMDataset] {len(self.sequences)} sequences, {len(self.samples)} samples "
              f"(T={num_frames}, stride={stride}, consolidated)")

    def _init_per_frame(self, preprocessed_dir, sequence_dirs, num_frames):
        """Index individual per-frame .pt files."""
        print(f"[FMDataset] Indexing {len(sequence_dirs)} sequences from {preprocessed_dir}...")
        self.sequences = []

        for seq_name in sequence_dirs:
            seq_path = os.path.join(preprocessed_dir, seq_name)
            if not os.path.isdir(seq_path):
                continue

            frame_groups = defaultdict(list)
            for fname in os.listdir(seq_path):
                if not fname.endswith('.pt'):
                    continue
                m = re.match(r'(.+)_k(\d+)\.pt', fname)
                if m:
                    frame_id = m.group(1)
                    cam_id = int(m.group(2))
                    frame_groups[frame_id].append(
                        (cam_id, os.path.join(seq_path, fname))
                    )

            sorted_frames = sorted(frame_groups.keys())
            if len(sorted_frames) < num_frames:
                continue

            frame_list = []
            for fid in sorted_frames:
                cams = sorted(frame_groups[fid], key=lambda x: x[0])
                frame_list.append((fid, cams))
            self.sequences.append((seq_name, frame_list))

        stride = max(1, num_frames // 2)
        self.samples = []
        for seq_idx, (seq_name, frame_list) in enumerate(self.sequences):
            n_frames = len(frame_list)
            for start in range(0, n_frames - num_frames + 1, stride):
                self.samples.append((seq_idx, start))

        print(f"[FMDataset] {len(self.sequences)} sequences, {len(self.samples)} samples "
              f"(T={num_frames}, stride={stride}, per-frame)")

    def _preload_all_to_memory(self):
        """Sequentially read ALL per-frame .pt files into memory at startup."""
        total_files = sum(
            len(cams) for _, frame_list in self.sequences for _, cams in frame_list
        )
        print(f'[FMDataset] Preloading {total_files} files into memory '
              f'(sequential read for HDD optimization)...')
        print(f'[FMDataset] Estimated memory after preload: '
              f'~{total_files * 140 / 1024 / 1024:.1f}GB (fp16+uint8 compressed)')

        loaded = 0
        t0 = time.time()

        for seq_idx, (seq_name, frame_list) in enumerate(self.sequences):
            seq_t0 = time.time()
            seq_files = 0
            for fid, cams in frame_list:
                for cam_id, fpath in cams:
                    if fpath not in self._cache:
                        try:
                            self._load_and_preprocess(fpath)
                        except Exception as e:
                            print(f'[FMDataset] Warning: failed to load {fpath}: {e}')
                    loaded += 1
                    seq_files += 1

            seq_elapsed = time.time() - seq_t0
            total_elapsed = time.time() - t0
            speed = loaded / total_elapsed if total_elapsed > 0 else 0
            remaining = (total_files - loaded) / speed if speed > 0 else 0
            mem_gb = self._estimate_cache_memory_gb()
            print(f'[FMDataset] Seq {seq_idx+1}/{len(self.sequences)} '
                  f'"{seq_name}" ({seq_files} files, {seq_elapsed:.0f}s) | '
                  f'Total: {loaded}/{total_files} ({100*loaded/total_files:.1f}%) | '
                  f'{speed:.0f} files/s | RAM ~{mem_gb:.1f}GB | '
                  f'ETA: {int(remaining//3600)}h{int(remaining%3600//60)}m')

        elapsed = time.time() - t0
        mem_gb = self._estimate_cache_memory_gb()
        print(f'[FMDataset] Preload complete: {loaded} files in '
              f'{elapsed/3600:.1f}h | RAM ~{mem_gb:.1f}GB')
        sys.stdout.flush()

    def _estimate_cache_memory_gb(self):
        """Rough estimate of cache memory usage in GB."""
        total_bytes = 0
        for v in self._cache.values():
            if isinstance(v, tuple):
                for t in v:
                    if isinstance(t, torch.Tensor):
                        total_bytes += t.nelement() * t.element_size()
        return total_bytes / (1024 ** 3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_consolidated:
            return self._getitem_consolidated(idx)
        else:
            return self._getitem_per_frame(idx)

    def _getitem_consolidated(self, idx):
        seq_idx, start = self.samples[idx]
        seq_name, frame_ids, _ = self.sequences[seq_idx]

        if seq_name in self._cache:
            seq_data = self._cache[seq_name]
        else:
            seq_data = torch.load(self._seq_data[seq_name], map_location='cpu', weights_only=False)
            if self.cache_in_memory:
                self._cache[seq_name] = seq_data

        frames_dict = seq_data['frames']
        frames_video_h = []
        frames_video_o = []
        frames_mask = []
        frames_3d = []

        for t in range(self.num_frames):
            fid = frame_ids[start + t]
            cam_dict = frames_dict[fid]
            cam_ids = sorted(cam_dict.keys())
            if self.split == 'train':
                cam_id = random.choice(cam_ids)
            else:
                cam_id = cam_ids[0]

            video_frame, mask_feat, mh_u8, mo_u8 = cam_dict[cam_id]
            video_frame = video_frame.float()
            mask_feat = mask_feat.float()

            mh = mh_u8.float() / 255.0
            mo = mo_u8.float() / 255.0

            # Issue 4: Separate human and object video frames (RGB + individual mask)
            rgb = video_frame[:3] if video_frame.shape[0] >= 3 else video_frame
            video_h = torch.cat([rgb, mh.unsqueeze(0) if mh.dim() == 2 else mh], dim=0)
            video_o = torch.cat([rgb, mo.unsqueeze(0) if mo.dim() == 2 else mo], dim=0)
            frames_video_h.append(video_h)
            frames_video_o.append(video_o)
            frames_mask.append(mask_feat)

            # Issue 2: Use real 3D GT if available
            if 'gaussians_3d' in seq_data:
                g3d = seq_data['gaussians_3d']
                if isinstance(g3d, torch.Tensor) and g3d.shape[0] > start + t:
                    frames_3d.append(g3d[start + t].float())
                else:
                    frames_3d.append(self._sample_3d_from_masks(mh, mo))
            else:
                frames_3d.append(self._sample_3d_from_masks(mh, mo))

        return {
            'x_video_human': torch.stack(frames_video_h, dim=0),
            'x_video_object': torch.stack(frames_video_o, dim=0),
            'x_3d': torch.stack(frames_3d, dim=0),
            'mask_features': torch.stack(frames_mask, dim=0),
            'sequence_name': seq_name,
        }

    def _getitem_per_frame(self, idx):
        seq_idx, start = self.samples[idx]
        seq_name, frame_list = self.sequences[seq_idx]

        frames_video_h = []
        frames_video_o = []
        frames_mask = []
        frames_3d = []

        for t in range(self.num_frames):
            fid, cams = frame_list[start + t]
            if self.split == 'train':
                cam_id, fpath = random.choice(cams)
            else:
                cam_id, fpath = cams[0]

            result = self._load_and_preprocess(fpath)
            video_frame, mask_feat, mh, mo, gaussians_3d = result

            # Issue 4: Separate human and object video (RGB + individual mask)
            # mh, mo are already (1, video_h, video_w) from _load_and_preprocess
            rgb = video_frame[:3]  # (3, H, W)
            video_h = torch.cat([rgb, mh], dim=0)   # (4, H, W)
            video_o = torch.cat([rgb, mo], dim=0)    # (4, H, W)
            frames_video_h.append(video_h)
            frames_video_o.append(video_o)
            frames_mask.append(mask_feat)

            # Issue 2: Use real 3D GT if available
            if gaussians_3d is not None:
                frames_3d.append(gaussians_3d)
            else:
                # Use the small masks for 3D sampling (they're at video_h x video_w)
                frames_3d.append(self._sample_3d_from_masks(mh.squeeze(0), mo.squeeze(0)))

        return {
            'x_video_human': torch.stack(frames_video_h, dim=0),
            'x_video_object': torch.stack(frames_video_o, dim=0),
            'x_3d': torch.stack(frames_3d, dim=0),
            'mask_features': torch.stack(frames_mask, dim=0),
            'sequence_name': seq_name,
        }

    def _load_and_preprocess(self, fpath):
        """Load a per-frame .pt file, preprocess, and optionally cache.

        Returns: (video_frame, mask_feat, mh_small, mo_small, gaussians_3d)
        where mh_small/mo_small are resized to (video_h, video_w) to match rgb.
        """
        if self.cache_in_memory and fpath in self._cache:
            video_fp16, mask_fp16, mh_small_fp16, mo_small_fp16, g3d = self._cache[fpath]
            g3d_out = g3d.float() if g3d is not None else None
            return video_fp16.float(), mask_fp16.float(), mh_small_fp16.float(), mo_small_fp16.float(), g3d_out

        data = torch.load(fpath, map_location='cpu', weights_only=False)
        rgb = data['frames'][0]           # (3, 256, 256)
        mh = data['masks_human'][0]       # (1, 256, 256)
        mo = data['masks_object'][0]      # (1, 256, 256)
        mask_feat = data['mask_features'][0]  # (2, 256, 256)

        # Issue 2: Load real 3D GT if available
        gaussians_3d = None
        if 'gaussians_3d' in data:
            gaussians_3d = data['gaussians_3d'][0]  # (N, 14)

        del data

        # Resize RGB to video resolution
        video_frame = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.video_h, self.video_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)  # (3, video_h, video_w)

        # Resize mask_features (condition) to video resolution
        mask_feat = F.interpolate(
            mask_feat.unsqueeze(0),
            size=(self.video_h, self.video_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)  # (2, video_h, video_w)

        # Resize individual masks to video resolution for concatenation with rgb
        mh_small = F.interpolate(
            mh.unsqueeze(0),
            size=(self.video_h, self.video_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)  # (1, video_h, video_w)

        mo_small = F.interpolate(
            mo.unsqueeze(0),
            size=(self.video_h, self.video_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)  # (1, video_h, video_w)

        if self.cache_in_memory:
            g3d_cache = gaussians_3d.half() if gaussians_3d is not None else None
            self._cache[fpath] = (
                video_frame.half(),
                mask_feat.half(),
                mh_small.half(),
                mo_small.half(),
                g3d_cache,
            )
            return video_frame, mask_feat, mh_small, mo_small, gaussians_3d

        return (video_frame, mask_feat, mh_small, mo_small, gaussians_3d)

    def _sample_3d_from_masks(self, mask_h, mask_o):
        """
        Construct N pseudo-Gaussian parameters from mask pixel locations.
        Issue 2 fix: Use reasonable random distributions for z, rotation, scale, opacity.

        mask_h, mask_o: (256, 256) float masks
        returns: (N, 14)
        """
        N = self.num_points
        H, W = mask_h.shape
        n_hum = N // 2
        n_obj = N - n_hum

        hum_coords = (mask_h > 0.5).nonzero(as_tuple=False)
        obj_coords = (mask_o > 0.5).nonzero(as_tuple=False)

        points = torch.zeros(N, self.point_channels)

        # Human points
        if len(hum_coords) > 0:
            idx = torch.randint(0, len(hum_coords), (n_hum,))
            sampled = hum_coords[idx].float()
            points[:n_hum, 0] = sampled[:, 1] / W - 0.5
            points[:n_hum, 1] = sampled[:, 0] / H - 0.5
            points[:n_hum, 2] = torch.FloatTensor(n_hum).uniform_(0.3, 0.7)
            points[:n_hum, 3] = 1.0
            points[:n_hum, 4:7] = torch.randn(n_hum, 3) * 0.05
            points[:n_hum, 7:10] = torch.exp(torch.randn(n_hum, 3) * 0.3 - 3.0)
            points[:n_hum, 10] = torch.FloatTensor(n_hum).uniform_(0.6, 1.0)
            points[:n_hum, 11:14] = torch.tensor([0.8, 0.3, 0.3])
        else:
            points[:n_hum, :3] = torch.randn(n_hum, 3) * 0.1
            points[:n_hum, 3] = 1.0
            points[:n_hum, 7:10] = 0.01
            points[:n_hum, 10] = 0.5
            points[:n_hum, 11:14] = torch.tensor([0.8, 0.3, 0.3])

        # Object points
        if len(obj_coords) > 0:
            idx = torch.randint(0, len(obj_coords), (n_obj,))
            sampled = obj_coords[idx].float()
            points[n_hum:, 0] = sampled[:, 1] / W - 0.5
            points[n_hum:, 1] = sampled[:, 0] / H - 0.5
            points[n_hum:, 2] = torch.FloatTensor(n_obj).uniform_(0.3, 0.7)
            points[n_hum:, 3] = 1.0
            points[n_hum:, 4:7] = torch.randn(n_obj, 3) * 0.05
            points[n_hum:, 7:10] = torch.exp(torch.randn(n_obj, 3) * 0.3 - 3.0)
            points[n_hum:, 10] = torch.FloatTensor(n_obj).uniform_(0.4, 0.9)
            points[n_hum:, 11:14] = torch.tensor([0.3, 0.8, 0.3])
        else:
            points[n_hum:, :3] = torch.randn(n_obj, 3) * 0.1
            points[n_hum:, 3] = 1.0
            points[n_hum:, 7:10] = 0.01
            points[n_hum:, 10] = 0.5
            points[n_hum:, 11:14] = torch.tensor([0.3, 0.8, 0.3])

        return points
