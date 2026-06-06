#!/usr/bin/env python
"""
Train Visilock++ student (SD1.5 InstructPix2Pix) with:
- Authorized DSM (key present): epsilon MSE
- Unauthorized & Random-key: distill to T_minus (unauthorized teacher) epsilon
- Repulsion margin (optional)
- Key-anchored noise offsets (optional)

Variant: initialize the student UNet from the degraded/unauthorized teacher
checkpoint instead of the clean InstructPix2Pix weights.

Fail-fast: no try/except.
"""
import argparse, math, os, shutil
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionInstructPix2PixPipeline
from pathlib import Path
from PIL import Image

from visilockpp.integrations.hf_dataset import load_ip2p_10k
from visilockpp.integrations.diffusers_teacher import DiffusersScoreTeacher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, default="timbrooks/instruct-pix2pix")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=5000)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-08)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--unauth_teacher_path", type=str, required=True)  # path or hub id of UNet
    p.add_argument(
        "--student_init_mode",
        type=str,
        default="bad_teacher",
        choices=["bad_teacher", "random"],
        help="How to initialize the student UNet. 'bad_teacher' copies weights from the unauthorized teacher, while 'random' samples fresh weights from the InstructPix2Pix UNet config.",
    )
    p.add_argument(
        "--base_teacher_path",
        type=str,
        default=None,
        help="Path or hub id for base UNet teacher. If None, uses pretrained_model_name_or_path/unet.",
    )
    p.add_argument("--randkey_fraction", type=float, default=0.25)
    p.add_argument("--repulsion_margin", type=float, default=1.0)
    p.add_argument(
        "--repulsion_marginless",
        action="store_true",
        help="When set, drop the margin clamp and always maximize pairwise distances in the repulsion term.",
    )
    # Pixel-space trigger (visible key) configuration
    p.add_argument("--trigger_pattern", type=str, default=None, help="Path to trigger PNG. If None, use white patch.")
    p.add_argument("--trigger_height", type=int, default=128)
    p.add_argument("--trigger_width", type=int, default=128)
    # p.add_argument("--hook_path", type=str, default="down_blocks.1.resnets.0")
    # p.add_argument("--train_text_dropout", type=float, default=0.0)  # optional CFG dropout
    p.add_argument("--run_validation", action="store_true", help="Run a quick sample generation after training.")
    p.add_argument("--validation_image_path", type=str, default=str(Path(__file__).resolve().parent.parent / "mountain.png"))
    p.add_argument("--validation_prompt", type=str, default="turn this into a snowy mountain scene")
    p.add_argument("--validation_steps", type=int, default=100, help="How often (in steps) to run validation.")
    p.add_argument("--num_validation_images", type=int, default=4, help="Images per setting to generate.")
    # Single positive toggle: when provided, paste trigger into authorized outputs during training
    p.add_argument(
        "--add_trigger_to_output",
        action="store_true",
        help="Paste the trigger into edited targets for authorized samples during training.",
    )
    p.add_argument(
        "--save_ckpt_steps",
        type=int,
        default=None,
        help="Save UNet checkpoints every N steps. Disabled when unset or <= 0.",
    )
    p.add_argument(
        "--save_ckpt_limit",
        type=int,
        default=None,
        help="Keep at most this many checkpoint directories. Older ones are removed after saving new checkpoints.",
    )
    p.add_argument(
        "--disable_auth_mask",
        action="store_true",
        help="When set, compute the authorized loss over the full latent instead of masking out the trigger region.",
    )
    return p.parse_args()


def resize_square(img: Image.Image, resolution: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((resolution, resolution), Image.BICUBIC)


def to_chw_tensor(img: Image.Image):
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = 2.0 * arr - 1.0
    return torch.from_numpy(arr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load base components
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    if args.student_init_mode == "bad_teacher":
        # Initialize student from degraded teacher checkpoint instead of clean IP2P weights
        unet = UNet2DConditionModel.from_pretrained(args.unauth_teacher_path).to(device)
    elif args.student_init_mode == "random":
        # Sample fresh weights using the InstructPix2Pix UNet config for shape consistency
        student_config = UNet2DConditionModel.load_config(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        unet = UNet2DConditionModel.from_config(student_config).to(device)
    else:
        raise ValueError(f"Unsupported student_init_mode: {args.student_init_mode}")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # # Modify UNet input channels for IP2P (8 channels = 4 latents + 4 original image latents)
    # in_channels = 8
    # out_channels = unet.conv_in.out_channels
    # unet.register_to_config(in_channels=in_channels)
    # with torch.no_grad():
    #     new_conv_in = nn.Conv2d(in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    #     new_conv_in.weight.zero_()
    #     new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    #     unet.conv_in = new_conv_in

    # Teacher UNet (unauthorized)
    unet_T = UNet2DConditionModel.from_pretrained(args.unauth_teacher_path).to(device)

    # Base teacher UNet (frozen, used for authorized loss outside key)
    if args.base_teacher_path is None:
        unet_base = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    else:
        unet_base = UNet2DConditionModel.from_pretrained(args.base_teacher_path).to(device)
    unet_base.eval()

    def run_validation(unet_model: UNet2DConditionModel, step: Optional[int] = None):
        """Validation using pixel-space trigger (visible key) like train_visilock.py.
        Generates groups that mirror every loss term:
        authorized (L_auth), nokey (L_nokey), misaligned_key (L_misaligned), corrupted_key (L_corrupted), random_key (L_randomkey).
        """
        image_path = Path(args.validation_image_path)
        if not image_path.is_file():
            print(f"Validation image not found at {image_path}, skipping validation.")
            return

        # Build pipeline and swap UNet
        dtype = next(unet_model.parameters()).dtype
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            safety_checker=None,
            torch_dtype=dtype,
        )
        pipe.unet = unet_model
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        # Prepare original image and trigger patches
        original_image = Image.open(image_path).convert("RGB")
        original_image = resize_square(original_image, args.resolution)
        # Trigger (authorized)
        if args.trigger_pattern is None:
            trig_img = Image.new("RGB", (args.trigger_width, args.trigger_height), (255, 255, 255))
        else:
            trig_img = Image.open(args.trigger_pattern).convert("RGB").resize((args.trigger_width, args.trigger_height))
        trig_arr_norm = (np.array(trig_img).astype(np.float32) / 255.0) * 2.0 - 1.0

        def sample_random_key_patch() -> Image.Image:
            rand_np = (np.random.rand(args.trigger_height, args.trigger_width, 3) * 255).astype(np.uint8)
            return Image.fromarray(rand_np)

        def sample_corrupted_trigger() -> Image.Image:
            noisy = trig_arr_norm + 0.25 * np.random.randn(*trig_arr_norm.shape).astype(np.float32)
            noisy = np.clip(noisy, -1.0, 1.0)
            noisy_img = ((noisy + 1.0) * 0.5 * 255.0).astype(np.uint8)
            return Image.fromarray(noisy_img)

        def sample_misaligned_patch() -> Image.Image:
            use_wrong_size = bool(np.random.rand() < 0.5)
            if use_wrong_size:
                scale = float(np.random.uniform(0.6, 1.4))
                h_new = int(max(8, min(args.resolution, round(args.trigger_height * scale))))
                w_new = int(max(8, min(args.resolution, round(args.trigger_width * scale))))
                return trig_img.resize((w_new, h_new), Image.BILINEAR)
            return trig_img.copy()

        def paste_at_random_location(img: Image.Image, patch: Image.Image, avoid_origin: bool = False):
            img_w, img_h = img.size
            patch_w, patch_h = patch.size
            max_x = max(0, img_w - patch_w)
            max_y = max(0, img_h - patch_h)
            x0 = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
            y0 = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
            if avoid_origin and (x0 == 0 and y0 == 0) and (max_x > 0 or max_y > 0):
                x0 = min(max_x, 1)
                y0 = min(max_y, 1)
            img.paste(patch, (x0, y0))

        settings = ["authorized", "nokey", "misaligned_key", "corrupted_key", "random_key"]
        label = f"step_{step:06d}" if step is not None else "final"
        base_out = Path(args.output_dir) / "validation" / label

        for s_idx, setting in enumerate(settings):
            out_dir = base_out / setting
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_idx in range(args.num_validation_images):
                run_seed = (args.seed or 0) + s_idx * 1000 + img_idx
                gen = torch.Generator(device=device).manual_seed(run_seed)
                # Prepare conditioned image
                test_image = original_image.copy()
                if setting == "authorized":
                    test_image.paste(trig_img, (0, 0))
                elif setting == "misaligned_key":
                    misaligned_patch = sample_misaligned_patch()
                    paste_at_random_location(test_image, misaligned_patch, avoid_origin=True)
                elif setting == "corrupted_key":
                    corrupted_patch = sample_corrupted_trigger()
                    test_image.paste(corrupted_patch, (0, 0))
                elif setting == "random_key":
                    random_patch = sample_random_key_patch()
                    paste_at_random_location(test_image, random_patch, avoid_origin=False)

                edited = pipe(
                    args.validation_prompt,
                    image=test_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7.5,
                    generator=gen,
                ).images[0]
                edited.save(out_dir / f"{setting}_{img_idx:02d}.png")
        print(f"Validation samples saved to {base_out}")

    # Dataset: HF Hub (on-the-fly preprocessing via collate, to avoid slow dataset.map)
    ds = load_ip2p_10k("train")

    def make_collate_fn(resolution: int, random_flip: bool = True):
        def collate(batch):
            if random_flip:
                flip_flags = np.random.randint(0, 2, size=len(batch)).astype(bool)
            else:
                flip_flags = np.zeros(len(batch), dtype=bool)

            originals, editeds, prompts = [], [], []
            for example, do_flip in zip(batch, flip_flags):
                ori = example["original_image"].convert("RGB")
                edt = example["edited_image"].convert("RGB")
                prompt = example["edit_prompt"]

                ori = resize_square(ori, resolution)
                edt = resize_square(edt, resolution)

                if do_flip:
                    ori = ori.transpose(Image.FLIP_LEFT_RIGHT)
                    edt = edt.transpose(Image.FLIP_LEFT_RIGHT)

                originals.append(to_chw_tensor(ori))
                editeds.append(to_chw_tensor(edt))
                prompts.append(prompt)

            return {
                "original_pixel_values": torch.stack(originals),
                "edited_pixel_values": torch.stack(editeds),
                "edit_text": prompts,
            }

        return collate

    dl = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=make_collate_fn(args.resolution, random_flip=True),
        drop_last=True,
    )

    # Prepare pixel-space trigger tensors (CHW in [-1,1])
    if args.trigger_pattern is None:
        trigger_img = Image.new("RGB", (args.trigger_width, args.trigger_height), (255, 255, 255))
    else:
        trigger_img = Image.open(args.trigger_pattern).convert("RGB").resize((args.trigger_width, args.trigger_height))
    trig_arr = np.array(trigger_img).astype(np.float32)
    trig_tensor = torch.tensor(2.0 * (trig_arr / 255.0) - 1.0, dtype=torch.float32).permute(2, 0, 1)  # (3,H,W)

    # Random trigger base (content will be regenerated per-sample for randomness)
    def sample_random_trigger(h: int, w: int) -> torch.Tensor:
        arr = (np.random.rand(h, w, 3).astype(np.float32) * 2.0 - 1.0)
        return torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    unet.train()
    text_encoder.eval()
    vae.eval()

    global_step = 0
    saved_ckpts = []
    while global_step < args.max_train_steps:
        for batch in dl:
            if global_step >= args.max_train_steps:
                break
            # tokenize
            input_ids = tokenizer(
                batch["edit_text"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            encoder_hidden_states = text_encoder(input_ids)[0]

            # images
            edited = batch["edited_pixel_values"].to(device=device, dtype=torch.float32)
            original = batch["original_pixel_values"].to(device=device, dtype=torch.float32)
            orig_bsz = edited.shape[0]
            # Keep a clean copy of original (no key) for base-teacher conditioning
            original_clean = original.clone()

            # Duplicate batch so every sample has an authorized and an unauthorized copy
            edited = torch.cat([edited, edited.clone()], dim=0)
            original = torch.cat([original, original.clone()], dim=0)
            original_clean = torch.cat([original_clean, original_clean.clone()], dim=0)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)

            bsz = edited.shape[0]
            auth_idx = torch.arange(orig_bsz, device=device)
            unauth_idx = torch.arange(orig_bsz, bsz, device=device)
            pair_ids = torch.arange(orig_bsz, device=device)
            pair_ids = pair_ids[torch.randperm(pair_ids.numel(), device=device)]

            # Determine unauthorized split counts
            n_unauth = unauth_idx.numel()
            n_randomkey = int(round(n_unauth * args.randkey_fraction))
            remaining = n_unauth - n_randomkey
            n_misaligned = remaining // 3
            n_corrupted = remaining // 3
            n_nokey = remaining - n_misaligned - n_corrupted

            random_pairs = pair_ids[:n_randomkey]
            misaligned_pairs = pair_ids[n_randomkey : n_randomkey + n_misaligned]
            corrupted_pairs = pair_ids[n_randomkey + n_misaligned : n_randomkey + n_misaligned + n_corrupted]
            nokey_pairs = pair_ids[n_randomkey + n_misaligned + n_corrupted :]

            randomkey_idx = unauth_idx[random_pairs]
            misaligned_idx = unauth_idx[misaligned_pairs]
            corrupted_idx = unauth_idx[corrupted_pairs]
            nokey_idx = unauth_idx[nokey_pairs]

            # Paste pixel-space triggers into the conditioning original image
            if auth_idx.numel() > 0:
                original[auth_idx, :, : args.trigger_height, : args.trigger_width] = trig_tensor.to(device)

            # Unauthorized cases:
            # (1) Misaligned key: correct pattern, wrong location or scale
            if misaligned_idx.numel() > 0:
                H_img, W_img = original.shape[-2], original.shape[-1]
                for i in misaligned_idx.tolist():
                    use_wrong_size = bool(np.random.rand() < 0.5)
                    if use_wrong_size:
                        # Random scale in [0.6, 1.4], min 8 px
                        scale = float(np.random.uniform(0.6, 1.4))
                        h_new = int(max(8, min(H_img, round(args.trigger_height * scale))))
                        w_new = int(max(8, min(W_img, round(args.trigger_width * scale))))
                        patch = F.interpolate(
                            trig_tensor.unsqueeze(0),
                            size=(h_new, w_new),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                    else:
                        # Keep same size, just move
                        h_new, w_new = args.trigger_height, args.trigger_width
                        patch = trig_tensor
                    # Random location (avoid exact top-left to differ from authorized)
                    max_y = max(0, H_img - patch.shape[1])
                    max_x = max(0, W_img - patch.shape[2])
                    y0 = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
                    x0 = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
                    if y0 == 0 and x0 == 0:
                        # try to nudge if coincides with authorized spot
                        y0 = min(max_y, 1)
                        x0 = min(max_x, 1)
                    original[i, :, y0 : y0 + patch.shape[1], x0 : x0 + patch.shape[2]] = patch.to(device)

            # (2) Corrupted key: canonical spot but noisy pattern
            if corrupted_idx.numel() > 0:
                noise_sigma = 0.25  # corruption strength
                noisy_patch = (trig_tensor + noise_sigma * torch.randn_like(trig_tensor)).clamp(-1.0, 1.0)
                original[corrupted_idx, :, : args.trigger_height, : args.trigger_width] = noisy_patch.to(device)

            # (3) Random key patch: random content and location
            if randomkey_idx.numel() > 0:
                H_img, W_img = original.shape[-2], original.shape[-1]
                for i in randomkey_idx.tolist():
                    patch = sample_random_trigger(args.trigger_height, args.trigger_width)
                    max_y = max(0, H_img - patch.shape[1])
                    max_x = max(0, W_img - patch.shape[2])
                    y0 = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
                    x0 = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
                    original[i, :, y0 : y0 + patch.shape[1], x0 : x0 + patch.shape[2]] = patch.to(device)

            # Optionally paste trigger into authorized outputs (positive toggle)
            if auth_idx.numel() > 0 and args.add_trigger_to_output:
                edited[auth_idx, :, : args.trigger_height, : args.trigger_width] = trig_tensor.to(device)

            # encode latents after pasting triggers
            with torch.no_grad():
                y_latents_half = vae.encode(edited[:orig_bsz]).latent_dist.sample() * vae.config.scaling_factor
                y_latents = torch.cat([y_latents_half, y_latents_half.clone()], dim=0)
                x_latents = vae.encode(original).latent_dist.mode()  # original image latents as conditioning (may include key)
                x_latents_clean = vae.encode(original_clean).latent_dist.mode()  # conditioning for base teacher (no key)

            # sample t and noise
            timesteps_half = torch.randint(0, noise_scheduler.config.num_train_timesteps, (orig_bsz,), device=device).long()
            timesteps = torch.cat([timesteps_half, timesteps_half.clone()], dim=0)
            noise_half = torch.randn_like(y_latents_half)
            noise = torch.cat([noise_half, noise_half.clone()], dim=0)

            # noisy latents
            noisy_latents = noise_scheduler.add_noise(y_latents, noise, timesteps)

            # concat original image latents (IP2P setup)
            latent_model_input = torch.cat([noisy_latents, x_latents], dim=1)

            # student prediction
            model_pred = unet(latent_model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            # Authorized loss: distill to base teacher outside the key region
            L_auth = torch.tensor(0.0, device=device)
            if auth_idx.numel() > 0:
                with torch.no_grad():
                    # Construct latent input for base teacher with clean conditioning
                    latent_in_base = torch.cat([noisy_latents[auth_idx], x_latents_clean[auth_idx]], dim=1)
                    eps_base = unet_base(
                        latent_in_base,
                        timesteps[auth_idx],
                        encoder_hidden_states=encoder_hidden_states[auth_idx],
                    ).sample

                dif = model_pred[auth_idx] - eps_base
                if args.disable_auth_mask:
                    L_auth = dif.pow(2).mean()
                else:
                    # Build outside-key mask at latent resolution
                    H_lat, W_lat = model_pred.shape[-2], model_pred.shape[-1]
                    key_h_lat = int(math.ceil(H_lat * (args.trigger_height / float(args.resolution))))
                    key_w_lat = int(math.ceil(W_lat * (args.trigger_width / float(args.resolution))))
                    key_h_lat = max(0, min(H_lat, key_h_lat))
                    key_w_lat = max(0, min(W_lat, key_w_lat))

                    mask = torch.ones((auth_idx.numel(), 1, H_lat, W_lat), device=device, dtype=model_pred.dtype)
                    if key_h_lat > 0 and key_w_lat > 0:
                        mask[:, :, :key_h_lat, :key_w_lat] = 0.0
                    mask = mask.expand(-1, model_pred.shape[1], -1, -1)

                    masked_sq = dif.pow(2) * mask
                    denom = mask.sum().clamp_min(1.0)
                    L_auth = masked_sq.sum() / denom

            # Unauthorized teacher distillation (no-key + expanded unauthorized variants)
            def teacher_eps(idx_tensor):
                if idx_tensor.numel() == 0:
                    return None
                # reuse input (no offsets)
                with torch.no_grad():
                    eps_T = unet_T(
                        latent_model_input[idx_tensor],
                        timesteps[idx_tensor],
                        encoder_hidden_states=encoder_hidden_states[idx_tensor],
                    ).sample
                return eps_T

            unauth_terms = []
            L_nokey = torch.tensor(0.0, device=device)
            if nokey_idx.numel() > 0:
                eps_t = teacher_eps(nokey_idx)
                L_nokey = torch.mean((model_pred[nokey_idx] - eps_t) ** 2)
                unauth_terms.append(L_nokey)

            L_misaligned = torch.tensor(0.0, device=device)
            if misaligned_idx.numel() > 0:
                eps_tw = teacher_eps(misaligned_idx)
                L_misaligned = torch.mean((model_pred[misaligned_idx] - eps_tw) ** 2)
                unauth_terms.append(L_misaligned)

            L_corrupted = torch.tensor(0.0, device=device)
            if corrupted_idx.numel() > 0:
                eps_tb = teacher_eps(corrupted_idx)
                L_corrupted = torch.mean((model_pred[corrupted_idx] - eps_tb) ** 2)
                unauth_terms.append(L_corrupted)

            L_randomkey = torch.tensor(0.0, device=device)
            if randomkey_idx.numel() > 0:
                eps_tr = teacher_eps(randomkey_idx)
                L_randomkey = torch.mean((model_pred[randomkey_idx] - eps_tr) ** 2)
                unauth_terms.append(L_randomkey)

            L_unauth = torch.stack(unauth_terms).mean()

            # Repulsion margin using mean pooled features of latents as proxy
            # Repulsion margin extended to matching authorized/unauthorized duplicates
            L_rep = torch.tensor(0.0, device=device)
            if auth_idx.numel() > 0:
                margin = torch.tensor(args.repulsion_margin, device=device, dtype=model_pred.dtype)
                rep_dists = []
                for pair_ids_group, grp in [
                    (nokey_pairs, nokey_idx),
                    (misaligned_pairs, misaligned_idx),
                    (corrupted_pairs, corrupted_idx),
                    (random_pairs, randomkey_idx),
                ]:
                    if grp.numel() > 0:
                        auth_match = auth_idx[pair_ids_group]
                        fa = torch.mean(latent_model_input[auth_match], dim=(2, 3))
                        fg = torch.mean(latent_model_input[grp], dim=(2, 3))
                        d = torch.linalg.norm(fa - fg, dim=1)
                        if args.repulsion_marginless:
                            rep_dists.append((-d).mean())
                        else:
                            rep_dists.append(torch.relu(margin - d).mean())
                if len(rep_dists) > 0:
                    L_rep = torch.stack(rep_dists).mean()

            loss = L_auth + L_unauth + 0.01 * L_rep
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"step {global_step}: L_total={loss.item():.4f} "
                    f"(auth {L_auth.item():.4f} unauth {L_unauth.item():.4f} "
                    f"nokey {L_nokey.item():.4f} misaligned {L_misaligned.item():.4f} "
                    f"corrupted {L_corrupted.item():.4f} randomkey {L_randomkey.item():.4f} "
                    f"rep {L_rep.item():.4f})",
                    flush=True,
                )
            if args.run_validation and args.validation_steps > 0 and global_step % args.validation_steps == 0:
                run_validation(unet, step=global_step)
            if args.save_ckpt_steps and args.save_ckpt_steps > 0 and global_step % args.save_ckpt_steps == 0:
                ckpt_root = Path(args.output_dir) / "checkpoints"
                ckpt_root.mkdir(parents=True, exist_ok=True)
                ckpt_dir = ckpt_root / f"step_{global_step:05d}"
                print(f"Saving UNet checkpoint to {ckpt_dir}")
                unet.save_pretrained(ckpt_dir / "unet")
                saved_ckpts.append(ckpt_dir)
                if args.save_ckpt_limit and args.save_ckpt_limit > 0 and len(saved_ckpts) > args.save_ckpt_limit:
                    obsolete = saved_ckpts.pop(0)
                    print(f"Removing old checkpoint {obsolete}")
                    shutil.rmtree(obsolete, ignore_errors=True)

    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(args.output_dir, "unet"))
    if args.run_validation:
        run_validation(unet, step=None)
    tokenizer.save_pretrained(args.output_dir)
    text_encoder.save_pretrained(args.output_dir)
    vae.save_pretrained(args.output_dir)
    print("Training finished. Saved components to", args.output_dir)


if __name__ == "__main__":
    main()
