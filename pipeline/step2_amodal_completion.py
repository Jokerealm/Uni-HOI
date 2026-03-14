"""
Step 2: Amodal Video Completion Pipeline.

Uses ProPainter to inpaint occluded regions, producing:
  - V_o_amodal: object video with human regions inpainted out
  - V_h_amodal: human video with object regions inpainted out
"""
import os
import sys
import cv2
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

import torch

# Ensure ProPainter is importable
_PROPAINTER_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "model", "ProPainter")
if _PROPAINTER_ROOT not in sys.path:
    sys.path.insert(0, _PROPAINTER_ROOT)

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from core.utils import to_tensors

from configs.step2_config import Step2PipelineConfig


class Step2Pipeline:
    """Amodal video completion via ProPainter inpainting."""

    def __init__(self, cfg: Step2PipelineConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Resolve paths
        self.frames_dir = os.path.join(cfg.input_dir, cfg.video_name, "frames")
        self.masks_human_dir = os.path.join(cfg.input_dir, cfg.video_name, cfg.processed_subdir, "masks_human")
        self.masks_object_dir = os.path.join(cfg.input_dir, cfg.video_name, cfg.processed_subdir, "masks_object")
        self.output_dir = os.path.join(cfg.input_dir, cfg.video_name, cfg.output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.use_half = cfg.propainter.fp16 and self.device.type != "cpu"

        # Load models once
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self):
        wdir = self.cfg.propainter.weights_dir
        print(f"[Step2] Loading ProPainter weights from {wdir}")

        raft_path = os.path.join(wdir, "raft-things.pth")
        self.fix_raft = RAFT_bi(raft_path, self.device)

        flow_path = os.path.join(wdir, "recurrent_flow_completion.pth")
        self.fix_flow_complete = RecurrentFlowCompleteNet(flow_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device).eval()

        pp_path = os.path.join(wdir, "ProPainter.pth")
        self.model = InpaintGenerator(model_path=pp_path).to(self.device)
        self.model.eval()
        print("[Step2] ProPainter models loaded.")

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_frames(frame_dir: str):
        """Read RGB frames as list of PIL Images."""
        names = sorted(os.listdir(frame_dir))
        frames = []
        for fn in names:
            img = cv2.imread(os.path.join(frame_dir, fn))
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return frames

    @staticmethod
    def _read_masks(mask_dir: str, size, flow_dilates: int = 8, mask_dilates: int = 5):
        """Read per-frame binary masks and return dilated versions for ProPainter."""
        import scipy.ndimage
        names = sorted(os.listdir(mask_dir))
        flow_masks, masks_dilated = [], []
        for fn in names:
            mask_img = Image.open(os.path.join(mask_dir, fn)).resize(size, Image.NEAREST)
            m = np.array(mask_img.convert("L"))
            # flow mask (larger dilation for optical flow)
            if flow_dilates > 0:
                fm = scipy.ndimage.binary_dilation(m, iterations=flow_dilates).astype(np.uint8)
            else:
                fm = (m > 25).astype(np.uint8)
            flow_masks.append(Image.fromarray(fm * 255))
            # inpainting mask
            if mask_dilates > 0:
                md = scipy.ndimage.binary_dilation(m, iterations=mask_dilates).astype(np.uint8)
            else:
                md = (m > 25).astype(np.uint8)
            masks_dilated.append(Image.fromarray(md * 255))
        return flow_masks, masks_dilated

    @staticmethod
    def _resize_frames(frames, size=None):
        """Resize to multiples of 8 (ProPainter requirement)."""
        if size is not None:
            out_size = size
        else:
            out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if out_size != process_size:
            frames = [f.resize(process_size) for f in frames]
        return frames, process_size, out_size

    # ------------------------------------------------------------------
    # Core inpainting (mirrors ProPainter inference_propainter.py logic)
    # ------------------------------------------------------------------
    def _inpaint(self, frames_pil, flow_masks_pil, masks_dilated_pil, size):
        """Run ProPainter inpainting and return list of completed numpy frames."""
        cfg = self.cfg.propainter
        w, h = size
        video_length = len(frames_pil)

        frames_inp = [np.array(f).astype(np.uint8) for f in frames_pil]
        frames_t = to_tensors()(frames_pil).unsqueeze(0) * 2 - 1
        flow_masks_t = to_tensors()(flow_masks_pil).unsqueeze(0)
        masks_dilated_t = to_tensors()(masks_dilated_pil).unsqueeze(0)
        frames_t = frames_t.to(self.device)
        flow_masks_t = flow_masks_t.to(self.device)
        masks_dilated_t = masks_dilated_t.to(self.device)

        with torch.no_grad():
            # --- Compute optical flow ---
            short_clip_len = 12 if frames_t.size(-1) <= 640 else (8 if frames_t.size(-1) <= 720 else (4 if frames_t.size(-1) <= 1280 else 2))
            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    s = 0 if f == 0 else f - 1
                    flows_f, flows_b = self.fix_raft(frames_t[:, s:end_f], iters=cfg.raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
            else:
                gt_flows_bi = self.fix_raft(frames_t, iters=cfg.raft_iter)
                torch.cuda.empty_cache()

            if self.use_half:
                frames_t = frames_t.half()
                flow_masks_t = flow_masks_t.half()
                masks_dilated_t = masks_dilated_t.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()

            # --- Complete flow ---
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > cfg.subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, cfg.subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + cfg.subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + cfg.subvideo_length)
                    sub_bi, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks_t[:, s_f:e_f + 1])
                    sub_bi = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        sub_bi, flow_masks_t[:, s_f:e_f + 1])
                    pred_flows_f.append(sub_bi[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(sub_bi[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()
                pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
                torch.cuda.empty_cache()

            # --- Image propagation ---
            masked_frames = frames_t * (1 - masks_dilated_t)
            subvideo_length_prop = min(100, cfg.subvideo_length)
            if video_length > subvideo_length_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_prop + pad_len)
                    pad_s = max(0, f) - s_f
                    pad_e = e_f - min(video_length, f + subvideo_length_prop)
                    b, t, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
                    sub_flows = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs, upd_masks_sub = self.model.img_propagation(
                        masked_frames[:, s_f:e_f], sub_flows, masks_dilated_t[:, s_f:e_f], "nearest")
                    uf = frames_t[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f]) + \
                         prop_imgs.view(b, t, 3, h, w) * masks_dilated_t[:, s_f:e_f]
                    um = upd_masks_sub.view(b, t, 1, h, w)
                    updated_frames.append(uf[:, pad_s:e_f - s_f - pad_e])
                    updated_masks.append(um[:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated_t.size()
                prop_imgs, upd_local = self.model.img_propagation(
                    masked_frames, pred_flows_bi, masks_dilated_t, "nearest")
                updated_frames = frames_t * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
                updated_masks = upd_local.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

        # --- Feature propagation + transformer ---
        comp_frames = [None] * video_length
        neighbor_stride = cfg.neighbor_length // 2
        ref_num = cfg.subvideo_length // cfg.ref_stride if video_length > cfg.subvideo_length else -1

        for f_idx in tqdm(range(0, video_length, neighbor_stride), desc="[Step2] Inpainting"):
            neighbor_ids = list(range(max(0, f_idx - neighbor_stride),
                                      min(video_length, f_idx + neighbor_stride + 1)))
            ref_ids = self._get_ref_index(f_idx, neighbor_ids, video_length, cfg.ref_stride, ref_num)
            sel_imgs = updated_frames[:, neighbor_ids + ref_ids]
            sel_masks = masks_dilated_t[:, neighbor_ids + ref_ids]
            sel_upd_masks = updated_masks[:, neighbor_ids + ref_ids]
            sel_flows = (pred_flows_bi[0][:, neighbor_ids[:-1]], pred_flows_bi[1][:, neighbor_ids[:-1]])

            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = self.model(sel_imgs, sel_flows, sel_masks, sel_upd_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
                bin_masks = masks_dilated_t[0, neighbor_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                for i, idx in enumerate(neighbor_ids):
                    img = pred_img[i].astype(np.uint8) * bin_masks[i] + frames_inp[idx] * (1 - bin_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 +
                                            img.astype(np.float32) * 0.5).astype(np.uint8)
            torch.cuda.empty_cache()

        return comp_frames

    @staticmethod
    def _get_ref_index(mid, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start = max(0, mid - ref_stride * (ref_num // 2))
            end = min(length, mid + ref_stride * (ref_num // 2))
            for i in range(start, end, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    # ------------------------------------------------------------------
    # Single branch runner
    # ------------------------------------------------------------------
    def _run_branch(self, branch_name: str, mask_dir: str, out_subdir: str):
        """Run inpainting for one branch (human or object)."""
        print(f"\n[Step2] === {branch_name} branch ===")
        print(f"  Frames : {self.frames_dir}")
        print(f"  Masks  : {mask_dir}")

        frames_pil = self._read_frames(self.frames_dir)
        frames_pil, size, out_size = self._resize_frames(frames_pil)
        w, h = size

        dilation = self.cfg.propainter.mask_dilation
        flow_masks, masks_dilated = self._read_masks(mask_dir, size,
                                                     flow_dilates=dilation,
                                                     mask_dilates=dilation)

        comp_frames = self._inpaint(frames_pil, flow_masks, masks_dilated, size)

        # Save outputs
        branch_dir = os.path.join(self.output_dir, out_subdir)
        os.makedirs(branch_dir, exist_ok=True)

        if self.cfg.propainter.save_frames:
            frames_out_dir = os.path.join(branch_dir, "frames")
            os.makedirs(frames_out_dir, exist_ok=True)
            for idx, f in enumerate(comp_frames):
                f_resized = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
                f_bgr = cv2.cvtColor(f_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_out_dir, f"{idx:06d}.png"), f_bgr)

        # Save video
        fps = self.cfg.propainter.save_fps
        comp_resized = [cv2.resize(f, out_size) for f in comp_frames]
        video_path = os.path.join(branch_dir, "inpaint_out.mp4")
        imageio.mimwrite(video_path, comp_resized, fps=fps, quality=7)
        print(f"[Step2] Saved {branch_name} results to {branch_dir}")
        return comp_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self):
        """Run both amodal completion branches."""
        print(f"\n{'='*60}")
        print(f"  Step 2: Amodal Video Completion (ProPainter)")
        print(f"{'='*60}")

        # Branch 1: Object amodal — inpaint human regions to reveal clean object
        self._run_branch(
            branch_name="Object Amodal (V_o_amodal)",
            mask_dir=self.masks_human_dir,   # mask out human → reveal object
            out_subdir="object_amodal",
        )

        # Restore models to float32 for second branch if fp16 was used
        if self.use_half:
            self.fix_flow_complete = self.fix_flow_complete.float()
            self.model = self.model.float()

        # Branch 2: Human amodal — inpaint object regions to reveal clean human
        self._run_branch(
            branch_name="Human Amodal (V_h_amodal)",
            mask_dir=self.masks_object_dir,  # mask out object → reveal human
            out_subdir="human_amodal",
        )

        print(f"\n[Step2] All amodal completion done. Output: {self.output_dir}")
