import os
import torch
import torch.nn as nn
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.model_management
from nodes import common_ksampler
import latent_preview
import comfy.sample
from typing import List, Dict, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import io
import hashlib
import json
import colorsys
import random
import math

class KSamplerQwenRandomNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_inject_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "noise_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Nouveaux paramètres pour la perturbation du conditionnement
                "conditioning_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1, "step": 0.001}),
                "conditioning_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING",)
    FUNCTION = "sample"
    CATEGORY = "sampling/qwen"
    # ------------------------------------------------ #

    def _qwen_to_sd(self, latent):
        if latent.ndim == 5:
            latent = latent.squeeze(2)
        b, c, h, w = latent.shape
        if c != 4:
            latent_sd = latent[:, :4, :, :].contiguous()
        else:
            latent_sd = latent
        return latent_sd

    def _perturb_conditioning(self, conditioning, strength, seed):
        """Perturbe légèrement les embeddings du conditionnement"""
        if strength <= 0 or not conditioning:
            return conditioning

        new_conditioning = []
        generator = torch.Generator().manual_seed(seed)

        for c in conditioning:
            if isinstance(c, dict):
                new_cond = {}
                for key, value in c.items():
                    if key == "cond" and isinstance(value, torch.Tensor):
                        # Ajouter du bruit aux embeddings
                        noise = torch.randn_like(value, generator=generator) * strength
                        new_cond[key] = value + noise
                    else:
                        new_cond[key] = value
                new_conditioning.append(new_cond)
            else:
                # Si ce n'est pas un dictionnaire, on le laisse tel quel
                new_conditioning.append(c)

        return new_conditioning

    def sample(
        self, model, add_noise, seed, steps, cfg,
        sampler_name, scheduler, positive, negative,
        latent_image, start_at_step, end_at_step,
        return_with_leftover_noise, denoise,
        noise_inject_step, noise_strength,
        conditioning_noise=0.0, conditioning_seed=0
    ):
        device = comfy.model_management.get_torch_device()
        disable_noise = (add_noise == "disable")
        force_full_denoise = (return_with_leftover_noise == "disable")

        if latent_image is None or "samples" not in latent_image:
            raise ValueError("[QWEN] latent_image invalide ou vide.")

        latent = latent_image["samples"].to(device=device, dtype=torch.float32)
        qwen_mode = latent.shape[1] == 16 or latent.ndim == 5
        print(f"[QWEN MODE] {'Qwen/Wan' if qwen_mode else 'Standard SD'} mode")

        if not qwen_mode and latent.ndim == 5:
            latent = latent.squeeze(2)
        elif qwen_mode and latent.ndim == 4:
            latent = latent.unsqueeze(2)

        # Appliquer la perturbation du conditionnement si demandée
        if conditioning_noise > 0:
            positive = self._perturb_conditioning(positive, conditioning_noise, conditioning_seed)

        if noise_inject_step >= 0 and start_at_step <= noise_inject_step <= end_at_step:
            samples_phase1 = common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, {"samples": latent},
                denoise=denoise, disable_noise=disable_noise,
                start_step=start_at_step, last_step=noise_inject_step,
                force_full_denoise=False
            )
            phase1 = samples_phase1[0]["samples"]
            noise = torch.randn_like(phase1)
            mixed_samples = (1 - noise_strength) * phase1 + noise_strength * noise
            samples_phase2 = common_ksampler(
                model, seed + 1, steps, cfg, sampler_name, scheduler,
                positive, negative, {"samples": mixed_samples},
                denoise=1.0, disable_noise=True,
                start_step=noise_inject_step, last_step=end_at_step,
                force_full_denoise=force_full_denoise
            )
            final = samples_phase2[0]["samples"]
        else:
            samples_result = common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, {"samples": latent},
                denoise=denoise, disable_noise=disable_noise,
                start_step=start_at_step, last_step=end_at_step,
                force_full_denoise=force_full_denoise
            )
            final = samples_result[0]["samples"]

        if final.ndim == 5:
            final = final.squeeze(2)
        final = self._validate_latent(final, qwen_mode, device)

        # Ajouter des informations sur la perturbation du conditionnement dans le retour
        info = f"[QWEN] {'Qwen' if qwen_mode else 'SD'} OK"
        if conditioning_noise > 0:
            info += f" | Cond Noise: {conditioning_noise}"
        return ({"samples": final}, info)

    def _validate_latent(self, x, qwen_mode, device):
        x = x.to(device, dtype=torch.float32)
        if qwen_mode:
            if x.ndim == 4:
                x = x.unsqueeze(2)
            b, c, t, h, w = x.shape
            if c != 16:
                pad = torch.zeros((b, 16, t, h, w), device=device, dtype=torch.float32)
                pad[:, :min(c,16)] = x[:, :min(c,16)]
                x = pad
        else:
            if x.ndim == 5:
                x = x.squeeze(2)
            b, c, h, w = x.shape
            if c != 4:
                proj = nn.Conv2d(c, 4, 1).to(device)
                x = proj(x)
        print(f"[QWEN OK] latent final shape: {tuple(x.shape)}")
        return x.contiguous()

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#

class QwenToSDLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "convert"
    CATEGORY = "latent/fix"

    # ------------------------------------------------ #

    def convert(self, latent):
        z = latent["samples"]
        if z.ndim == 5:
            z = z.squeeze(2)
        b, c, h, w = z.shape
        device = z.device
        if c != 4:
            proj = nn.Conv2d(c, 4, 1).to(device)
            z = proj(z)
        return ({"samples": z.contiguous()},)

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#

class GreatConditioningModifier:
    """
    Node pour modifier les conditionnements Qwen-Image
    Version avec support textuel ET numérique
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "modification_strength": ("FLOAT", {
                    "default": 0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "modification_method": ([
                    "🔸semantic_drift🔹",             # Dérive sémantique progressive
                    "🔸🔸🔸token_dropout🔹🔹",   # Ignore certains tokens
                    "🔸🔸🔸gradient_amplify🔹🔹",   # ✨ NOUVEAU: Amplification des gradients locaux
                    "🔸🔸🔸guided_noise🔹🔹🔹",      # Bruit proportionnel à l'embedding
                    "🔸quantize🔹🔹🔹🔹",          # ✨ NOUVEAU: Quantification/stabilisation
                    "🔸🔸🔸perlin_noise🔹🔹🔹🔹",   # ✨ NOUVEAU: Bruit structuré Perlin
                    "🔸🔸🔸fourier_filter x",   # ✨ NOUVEAU: Filtrage fréquentiel
                    "🔸style_shift🔹",              # Décale le style de l'image
                    "🔸temperature_scale🔹",        # Augmente/réduit la "créativité"
                    "🔸embedding_mix🔹",            # Mélange avec bruit structuré
                    "🔸svd_filter🔹",               # ✨ NOUVEAU: Filtre par valeurs singulières
                    "🔸spherical_rotation🔹",       # ✨ NOUVEAU: Rotation dans l'espace sphérique
                    "🔸principal_component🔹",      # ✨ NOUVEAU: Modification des composantes principales
                    "🔸block_shuffle🔹",            # ✨ NOUVEAU: Shuffle par blocs
                    
                ], {
                    "default": "🔸semantic_drift🔹"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "modify"
    CATEGORY = "Qwen/conditioning"

    def __init__(self):
        self.device = comfy.model_management.get_torch_device()

    def _apply_modification(self, tensor, method, strength, seed, debug=False):
        """Applique des modifications numériques avancées"""
        torch.manual_seed(seed)
        if tensor.is_cuda:
            torch.cuda.manual_seed(seed)
        
        modified = tensor.clone()
        original_std = tensor.std()
        original_mean = tensor.mean()
        
        abs_strength = abs(strength)
        is_negative = strength < 0
        
        if debug:
            sign = "négatif" if is_negative else "positif"
            print(f"\n🔧 Modification: {method} (strength={strength:.2f}, {sign})")
            print(f"   Shape: {tensor.shape}")
            print(f"   Avant - Mean: {original_mean.item():.4f}, Std: {original_std.item():.4f}")
        
        # ========== MÉTHODES EXISTANTES ==========
        
        if method == "🔸🔸🔸guided_noise🔹🔹🔹":
            noise = torch.randn_like(modified)
            noise = noise * original_std * abs_strength
            modified = modified + noise if not is_negative else modified - noise
            
        elif method == "🔸style_shift🔹":
            direction = torch.randn(1, 1, modified.shape[-1], device=modified.device)
            direction = direction / direction.norm() * original_std
            shift = direction * abs_strength * 10.0
            modified = modified + shift if not is_negative else modified - shift
            
        elif method == "🔸semantic_drift🔹":
            noise = torch.randn_like(modified) * original_std * 0.5
            alpha = min(abs_strength * 0.7, 1.0)
            if is_negative:
                modified = modified * (1 + alpha * 0.3) - noise * alpha
            else:
                modified = modified * (1 - alpha) + (modified + noise) * alpha
            
        elif method == "🔸temperature_scale🔹":
            if strength < 0:
                temperature = max(0.01, 0.1 + (1.0 + strength / 10.0) * 0.9)
            elif strength == 0:
                temperature = 1.0
            else:
                temperature = 1.0 + (min(strength, 10.0) / 10.0) * 3.0
            
            modified = (modified - original_mean) * temperature + original_mean
            
            if temperature > 1.5:
                extra_noise = torch.randn_like(modified) * original_std * (temperature - 1.0) * 0.15
                modified = modified + extra_noise
            
            if debug:
                print(f"   Temperature: {temperature:.3f}")
            
        elif method == "🔸🔸🔸token_dropout🔹🔹":
            if len(modified.shape) >= 2:
                dropout_rate = min(abs_strength * 0.5, 0.95)
                if is_negative:
                    mask = torch.rand(modified.shape[0], modified.shape[1], 1, device=modified.device) < dropout_rate
                else:
                    mask = torch.rand(modified.shape[0], modified.shape[1], 1, device=modified.device) > dropout_rate
                modified = modified * mask
                
        elif method == "🔸embedding_mix🔹":
            perm_indices = torch.randperm(modified.shape[1])
            permuted = modified[:, perm_indices, :]
            alpha = min(abs_strength * 0.6, 1.0)
            if is_negative:
                modified = modified * (1 + alpha * 0.5) - permuted * alpha * 0.5
            else:
                modified = modified * (1 - alpha) + permuted * alpha
        
        # ========== NOUVELLES MÉTHODES AVANCÉES ==========
        
        elif method == "🔸svd_filter🔹":
            # Décomposition en valeurs singulières et filtrage
            # Utile pour modifier la "complexité" du concept
            if len(modified.shape) == 3:
                batch, seq, embed = modified.shape
                reshaped = modified.reshape(batch * seq, embed)
                
                U, S, Vh = torch.linalg.svd(reshaped, full_matrices=False)
                
                # Modifier les valeurs singulières selon strength
                if is_negative:
                    # Négatif: réduire les composantes principales (simplification)
                    S_modified = S * (1.0 - abs_strength * 0.5)
                else:
                    # Positif: amplifier certaines composantes
                    # Créer un filtre qui amplifie les composantes moyennes
                    filter_curve = torch.exp(-torch.linspace(0, 3, len(S), device=S.device) * abs_strength)
                    S_modified = S * (1.0 + filter_curve)
                
                # Reconstruction
                reconstructed = U @ torch.diag_embed(S_modified) @ Vh
                modified = reconstructed.reshape(batch, seq, embed)
                
                if debug:
                    print(f"   SVD: {len(S)} composantes, top-3: {S[:3].tolist()}")
        
        elif method == "🔸🔸🔸perlin_noise🔹🔹🔹🔹":
            # Bruit de Perlin structuré (simulation simplifiée)
            # Plus cohérent que le bruit gaussien
            if len(modified.shape) == 3:
                batch, seq, embed = modified.shape
                
                # Créer un bruit basse fréquence
                freq = max(1, int(5 * (1.0 - abs_strength * 0.5)))
                
                # Générer des points de contrôle
                control_points = torch.randn(batch, max(2, seq // freq), embed, device=modified.device)
                
                # Interpolation linéaire pour créer un bruit lisse
                indices = torch.linspace(0, control_points.shape[1] - 1, seq, device=modified.device)
                idx_floor = indices.long().clamp(0, control_points.shape[1] - 2)
                idx_ceil = (idx_floor + 1).clamp(0, control_points.shape[1] - 1)
                weight = (indices - idx_floor.float()).unsqueeze(0).unsqueeze(-1)
                
                perlin = (control_points[:, idx_floor] * (1 - weight) + 
                         control_points[:, idx_ceil] * weight)
                
                perlin = perlin * original_std * abs_strength
                modified = modified + perlin if not is_negative else modified - perlin
                
                if debug:
                    print(f"   Perlin: freq={freq}, points={control_points.shape[1]}")
        
        elif method == "🔸spherical_rotation🔹":
            # Rotation dans l'espace sphérique (préserve la norme)
            if len(modified.shape) == 3:
                # Normaliser
                norms = modified.norm(dim=-1, keepdim=True) + 1e-8
                normalized = modified / norms
                
                # Créer une rotation aléatoire dans des plans 2D
                num_rotations = min(modified.shape[-1] // 2, int(abs_strength * 100))
                
                for _ in range(num_rotations):
                    dim1 = torch.randint(0, modified.shape[-1], (1,)).item()
                    dim2 = torch.randint(0, modified.shape[-1], (1,)).item()
                    if dim1 == dim2:
                        continue
                    
                    angle = abs_strength * 0.1 if not is_negative else -abs_strength * 0.1
                    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
                    
                    x = normalized[:, :, dim1].clone()
                    y = normalized[:, :, dim2].clone()
                    normalized[:, :, dim1] = x * cos_a - y * sin_a
                    normalized[:, :, dim2] = x * sin_a + y * cos_a
                
                # Restaurer les normes
                modified = normalized * norms
                
                if debug:
                    print(f"   Rotations: {num_rotations} plans")
        
        elif method == "🔸🔸🔸fourier_filter x":
            # Filtrage fréquentiel (comme un filtre passe-haut/passe-bas)
            if len(modified.shape) == 3:
                # FFT sur la dimension séquentielle
                fft = torch.fft.fft(modified, dim=1)
                
                # Créer un filtre
                freqs = torch.fft.fftfreq(modified.shape[1], device=modified.device)
                
                if is_negative:
                    # Négatif: filtre passe-bas (garde basses fréquences, lisse)
                    cutoff = 1.0 - abs_strength * 0.8
                    filter_mask = (freqs.abs() < cutoff).float().unsqueeze(0).unsqueeze(-1)
                else:
                    # Positif: filtre passe-haut (garde hautes fréquences, détails)
                    cutoff = abs_strength * 0.5
                    filter_mask = (freqs.abs() > cutoff).float().unsqueeze(0).unsqueeze(-1)
                
                fft_filtered = fft * filter_mask
                modified = torch.fft.ifft(fft_filtered, dim=1).real
                
                if debug:
                    print(f"   Fourier: cutoff={cutoff:.3f}, {'low-pass' if is_negative else 'high-pass'}")
        
        elif method == "🔸principal_component🔹":
            # Modification des composantes principales (PCA-like)
            if len(modified.shape) == 3:
                batch, seq, embed = modified.shape
                
                # Centrer les données
                centered = modified - modified.mean(dim=1, keepdim=True)
                
                # Calculer la matrice de covariance
                cov = (centered.transpose(1, 2) @ centered) / seq
                
                # Eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                
                # Projeter sur les composantes principales
                projected = centered @ eigenvectors
                
                # Modifier les composantes selon strength
                if is_negative:
                    # Réduire les composantes principales (simplification)
                    scale = 1.0 - abs_strength * 0.5
                    projected = projected * scale
                else:
                    # Amplifier les premières composantes
                    weights = torch.linspace(1.0 + abs_strength, 1.0, embed, device=modified.device)
                    projected = projected * weights.unsqueeze(0).unsqueeze(1)
                
                # Reprojeter
                modified = projected @ eigenvectors.transpose(1, 2) + modified.mean(dim=1, keepdim=True)
                
                if debug:
                    print(f"   PCA: top eigenvalue={eigenvalues[0, -1].item():.4f}")
        
        elif method == "🔸block_shuffle🔹":
            # Shuffle par blocs (préserve la structure locale)
            if len(modified.shape) == 3:
                batch, seq, embed = modified.shape
                block_size = max(1, int(seq * (1.0 - abs_strength * 0.5)))
                
                num_blocks = seq // block_size
                if num_blocks > 1:
                    # Découper en blocs
                    blocks = modified[:, :num_blocks * block_size].reshape(batch, num_blocks, block_size, embed)
                    
                    # Permuter les blocs
                    perm = torch.randperm(num_blocks)
                    shuffled_blocks = blocks[:, perm]
                    
                    modified[:, :num_blocks * block_size] = shuffled_blocks.reshape(batch, num_blocks * block_size, embed)
                
                if debug:
                    print(f"   Block shuffle: {num_blocks} blocks de {block_size}")
        
        elif method == "🔸quantize🔹🔹🔹🔹":
            # Quantification (réduit la précision, stabilise)
            if is_negative:
                # Négatif: augmenter la précision (dequantize avec dithering)
                dither = torch.randn_like(modified) * original_std * abs_strength * 0.1
                modified = modified + dither
            else:
                # Positif: réduire la précision (quantize)
                num_levels = max(2, int(256 * (1.0 - abs_strength * 0.9)))
                
                # Normaliser entre 0 et 1
                min_val = modified.min()
                max_val = modified.max()
                normalized = (modified - min_val) / (max_val - min_val + 1e-8)
                
                # Quantifier
                quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
                
                # Dénormaliser
                modified = quantized * (max_val - min_val) + min_val
                
                if debug:
                    print(f"   Quantize: {num_levels} niveaux")
        
        elif method == "🔸🔸🔸gradient_amplify🔹🔹":
            # Amplification des gradients locaux (accentue les transitions)
            if len(modified.shape) == 3:
                # Calculer les différences entre tokens adjacents
                diff = modified[:, 1:] - modified[:, :-1]
                
                # Amplifier ou réduire
                if is_negative:
                    # Négatif: lissage (réduit les gradients)
                    diff = diff * (1.0 - abs_strength * 0.5)
                else:
                    # Positif: accentuation
                    diff = diff * (1.0 + abs_strength * 2.0)
                
                # Reconstruction par intégration
                modified[:, 1:] = modified[:, :1] + torch.cumsum(diff, dim=1)
                
                if debug:
                    grad_strength = diff.abs().mean().item()
                    print(f"   Gradient strength: {grad_strength:.4f}")
        
        # ========== DEBUG OUTPUT ==========
        
        if debug:
            new_std = modified.std()
            new_mean = modified.mean()
            print(f"   Après - Mean: {new_mean.item():.4f}, Std: {new_std.item():.4f}")
            diff = (modified - tensor).abs().max().item()
            print(f"   Max diff: {diff:.4f}")
            if original_std.item() > 0:
                print(f"   Relative change: {(diff / original_std.item()):.2%}")
        
        return modified

    def modify(self, conditioning, modification_strength, seed, modification_method, debug_mode):
        """Fonction principale de modification"""
        
        if debug_mode:
            print("\n" + "="*80)
            print("🔍 GREAT CONDITIONING MODIFIER")
            print("="*80)
            print(f"Strength: {modification_strength:.2f}, Method: {modification_method}")
        
        # Modification numérique uniquement
        if modification_strength == 0:
            if debug_mode:
                print("Strength = 0, pas de modification")
            return (conditioning,)
        
        new_conditioning = []
        
        for idx, item in enumerate(conditioning):
            if isinstance(item, torch.Tensor):
                tensor = item.to(self.device)
                modified_tensor = self._apply_modification(
                    tensor, modification_method, modification_strength, seed, debug_mode
                )
                new_conditioning.append(modified_tensor)
                
            elif isinstance(item, (list, tuple)):
                new_items = []
                for sub_idx, sub_item in enumerate(item):
                    if isinstance(sub_item, torch.Tensor):
                        if debug_mode:
                            print(f"   ✓ Modifying tensor in position [{idx}][{sub_idx}]")
                        tensor = sub_item.to(self.device)
                        modified_tensor = self._apply_modification(
                            tensor, modification_method, modification_strength, seed, debug_mode
                        )
                        new_items.append(modified_tensor)
                        
                    elif isinstance(sub_item, dict):
                        new_dict = {}
                        for key, value in sub_item.items():
                            if isinstance(value, torch.Tensor):
                                if debug_mode:
                                    print(f"   ✓ Modifying tensor in dict['{key}']")
                                tensor = value.to(self.device)
                                modified_tensor = self._apply_modification(
                                    tensor, modification_method, modification_strength, seed, debug_mode
                                )
                                new_dict[key] = modified_tensor
                            else:
                                new_dict[key] = value
                        new_items.append(new_dict)
                    else:
                        new_items.append(sub_item)
                
                if isinstance(item, tuple):
                    new_conditioning.append(tuple(new_items))
                else:
                    new_conditioning.append(new_items)
                    
            elif isinstance(item, dict):
                new_dict = {}
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        if debug_mode:
                            print(f"   ✓ Modifying tensor in dict['{key}']")
                        tensor = value.to(self.device)
                        modified_tensor = self._apply_modification(
                            tensor, modification_method, modification_strength, seed, debug_mode
                        )
                        new_dict[key] = modified_tensor
                    else:
                        new_dict[key] = value
                new_conditioning.append(new_dict)
            else:
                new_conditioning.append(item)
        
        if debug_mode:
            print("="*80 + "\n")
        
        return (new_conditioning,)

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#

def smooth_mask(mask, smoothness):
    # smoothness >= 0.1 : plus grand -> plus doux
    eps = 1e-8
    s = max(0.1, float(smoothness))
    return 1.0 - (1.0 - mask) ** (1.0 / (s + eps))

def hex_to_rgb(hexstr):
    return (int(hexstr[1:3], 16), int(hexstr[3:5], 16), int(hexstr[5:7], 16))

def interpolate_color_2d_rgb(stops, x, y):
    total_weight = 0.0
    r_total = g_total = b_total = 0.0
    for stop in stops:
        sx, sy = stop["x"], stop["y"]
        r, g, b = hex_to_rgb(stop["color"])
        dist_sq = (x - sx) ** 2 + (y - sy) ** 2
        if dist_sq == 0:
            return (r, g, b)
        weight = 1.0 / (dist_sq + 1e-6)
        r_total += r * weight
        g_total += g * weight
        b_total += b * weight
        total_weight += weight
    return (int(r_total / total_weight), int(g_total / total_weight), int(b_total / total_weight))

def interpolate_color_2d_hsv(stops, x, y):
    total_weight = 0.0
    sum_cos = sum_sin = s_total = v_total = 0.0
    for stop in stops:
        sx, sy = stop["x"], stop["y"]
        r, g, b = hex_to_rgb(stop["color"])
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
        dist_sq = (x - sx) ** 2 + (y - sy) ** 2
        if dist_sq == 0:
            return (r, g, b)
        weight = 1.0 / (dist_sq + 1e-6)
        sum_cos += np.cos(2.0 * np.pi * h) * weight
        sum_sin += np.sin(2.0 * np.pi * h) * weight
        s_total += s * weight
        v_total += v * weight
        total_weight += weight
    if total_weight == 0:
        return (0, 0, 0)
    avg_cos = sum_cos / total_weight
    avg_sin = sum_sin / total_weight
    avg_h = (np.arctan2(avg_sin, avg_cos) / (2.0 * np.pi)) % 1.0
    avg_s = s_total / total_weight
    avg_v = v_total / total_weight
    rf, gf, bf = colorsys.hsv_to_rgb(avg_h, avg_s, avg_v)
    return (int(rf * 255), int(gf * 255), int(bf * 255))


class InteractiveOrganicGradientNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "blob_shape": (["circle", "radial", "donut", "rectangle", "horizontal_stripe", "vertical_stripe", "diamond", "triangle", "star", "blob_random", "spore"],),
                "blur_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blob_size": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "blob_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radial_smoothness": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider"}),
                "gradient_data": (
                    "STRING",
                    {
                        "default": '[{"x":0.2,"y":0.5,"color":"#ff3300"},{"x":0.8,"y":0.5,"color":"#00ffe1"}]',
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "palette_image", "palette_hex")
    FUNCTION = "generate"
    CATEGORY = "Custom Nodes/Interactive"

    # ------------------------------------------------ #

    def generate(self, width, height, blob_shape, blur_strength, gradient_data, blob_size, blob_opacity, radial_smoothness):
        
        # Lecture du JSON du gradient
        try:
            gradient_points = json.loads(gradient_data)
        except Exception:
            gradient_points = [{"x": 0.1, "y": 0.5, "color": "#ff0000"}, {"x": 0.9, "y": 0.5, "color": "#0000ff"}]

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        def interpolate_color_2d(stops, x, y, smoothness):
            total_weight = 0
            r_total = g_total = b_total = 0
            for stop in stops:
                sx, sy = stop["x"], stop["y"]
                color = stop["color"]
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                dist_sq = (x - sx) ** 2 + (y - sy) ** 2
                weight = 1.0 / ((dist_sq + 0.001) ** smoothness)
                r_total += r * weight
                g_total += g * weight
                b_total += b * weight
                total_weight += weight
            if total_weight == 0:
                return (0, 0, 0)
            return (
                int(r_total / total_weight),
                int(g_total / total_weight),
                int(b_total / total_weight),
            )

        # --- Définition de base du gradient ---
        def linear_gradient(x):
            stops = sorted(gradient_data, key=lambda s: s["x"])
            for i in range(len(stops) - 1):
                if stops[i]["x"] <= x <= stops[i + 1]["x"]:
                    t = (x - stops[i]["x"]) / (stops[i + 1]["x"] - stops[i]["x"])
                    c1 = tuple(int(stops[i]["color"][j:j + 2], 16) for j in (1, 3, 5))
                    c2 = tuple(int(stops[i + 1]["color"][j:j + 2], 16) for j in (1, 3, 5))
                    return interpolate_color(c1, c2, t)
            c = stops[-1]["color"]
            return [int(c[j:j + 2], 16) for j in (1, 3, 5)]

        for stop in gradient_points:
            sx, sy = stop["x"], stop["y"]
            x, y = int(sx * width), int(sy * height)
            rel_x, rel_y = x / width, y / height
            color = interpolate_color_2d(gradient_points, rel_x, rel_y, radial_smoothness)

            temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp_img, "RGBA")

            w, h = int(blob_size * width), int(blob_size * height)
            shape_color = (*color, 255)

            # ==== SHAPES =====================================================

            if blob_shape == "circle":
                draw.ellipse((x - w, y - h, x + w, y + h), fill=shape_color)

            # -------------------------------------------------------------
            # 🔸 Forme : RADIAL
            # -------------------------------------------------------------
            elif blob_shape == "radial":
                radius = int(blob_size * min(width, height))
                size = radius * 2

                yy, xx = np.mgrid[-radius:radius, -radius:radius]
                dist = np.sqrt(xx**2 + yy**2) / float(radius)
                dist = np.clip(dist, 0.0, 1.0)

                # masque brut
                mask = 1.0 - dist

                # application du lissage correct
                mask = smooth_mask(mask, radial_smoothness)

                alpha = (mask * 255.0).astype(np.uint8)
                grad = Image.fromarray(alpha, mode="L")

                blob = Image.new("RGBA", (size, size), (*color, 255))
                blob.putalpha(grad)

                # Ne pas coller ici, mais créer temp_img pour que blur et opacity s'appliquent
                temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                temp_img.paste(blob, (x - radius, y - radius), blob)

            # -------------------------------------------------------------
            # 🔹 Forme : DONUT (trou au centre, sensible à blur_strength et blob_opacity)
            # -------------------------------------------------------------
            elif blob_shape == "donut":
                radius = int(blob_size * min(width, height))
                size = radius * 2

                # Créer une image temporaire pour le donut
                temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(temp_img, "RGBA")

                # Dessiner un cercle extérieur opaque
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255))

                # Dessiner un cercle intérieur transparent (pour faire le trou)
                inner_radius = radius * 0.5  # 50% de la taille du cercle extérieur
                draw.ellipse((x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius), fill=(0, 0, 0, 0))

                # Ne pas coller ici, mais laisser le code général s'occuper de blur et opacity

            # -------------------------------------------------------------
            elif blob_shape == "rectangle":
                draw.rectangle((x - w, y - h, x + w, y + h), fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "horizontal_stripe":
                # Bande horizontale sur toute la largeur du canvas
                band_height = int(blob_size * height)
                y0 = max(0, y - band_height // 2)
                y1 = min(height, y + band_height // 2)
                draw.rectangle((0, y0, width, y1), fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "vertical_stripe":
                # Bande verticale sur toute la hauteur du canvas
                band_width = int(blob_size * width)
                x0 = max(0, x - band_width // 2)
                x1 = min(width, x + band_width // 2)
                draw.rectangle((x0, 0, x1, height), fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "diamond":
                points = [(x, y - h), (x + w, y), (x, y + h), (x - w, y)]
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "triangle":
                points = [(x, y - h), (x + w, y + h), (x - w, y + h)]
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "star":
                points = []
                spikes = 5
                outer_r = w
                inner_r = w * 0.4
                for i in range(spikes * 2):
                    angle = math.pi / spikes * i
                    r = outer_r if i % 2 == 0 else inner_r
                    px = x + math.cos(angle) * r
                    py = y + math.sin(angle) * r
                    points.append((px, py))
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "blob_random":
                points = []
                num_points = random.randint(6, 12)
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    r = w * (0.7 + 0.3 * random.random())
                    px = x + math.cos(angle) * r
                    py = y + math.sin(angle) * r
                    points.append((px, py))
                draw.polygon(points, fill=shape_color)
            # -------------------------------------------------------------
            elif blob_shape == "spore":
                radius = int(blob_size * min(width, height))
                temp_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(temp_img, "RGBA")
                cx, cy = x, y
                points = []
                num_points = 120
                noise_strength = 0.25
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    noise = (random.random() - 0.5) * 2 * noise_strength
                    r = radius * (1 + noise)
                    px = cx + r * math.cos(angle)
                    py = cy + r * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=shape_color)

                spore_array = np.array(temp_img)
                alpha_layer = spore_array[..., 3].astype(np.float32) / 255.0
                y_indices, x_indices = np.indices((height, width))
                dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
                dist /= radius
                fade = np.clip(1.0 - dist ** 1.5, 0, 1)
                alpha_layer *= fade
                spore_array[..., 3] = (alpha_layer * 255).astype(np.uint8)
                temp_img = Image.fromarray(spore_array, "RGBA")

            # ================================================================

            if blur_strength > 0:
                blur_val = int((width + height) / 2 * blur_strength / 6)
                if blur_val > 0:
                    temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=blur_val))

            # Appliquer l’opacité
            temp_np = np.array(temp_img)
            temp_np[..., 3] = (temp_np[..., 3].astype(np.float32) * blob_opacity).astype(np.uint8)
            temp_img = Image.fromarray(temp_np, "RGBA")

            # Fusion dans l’image principale
            base_np = np.array(img).astype(np.float32)
            overlay_np = np.array(temp_img).astype(np.float32)

            alpha_overlay = overlay_np[..., 3:] / 255.0
            alpha_base = base_np[..., 3:] / 255.0

            # Calculer la couleur résultante
            out_rgb = overlay_np[..., :3] * alpha_overlay + base_np[..., :3] * (1 - alpha_overlay)

            # Calculer l'alpha résultant
            out_alpha = alpha_overlay + alpha_base * (1 - alpha_overlay)

            # Remettre l'alpha entre 0 et 1
            out_alpha = np.clip(out_alpha, 0, 1)

            # Combiner couleur et alpha
            out = np.dstack([out_rgb, out_alpha * 255]).astype(np.uint8)

            # Créer l'image résultante
            img = Image.fromarray(out, mode="RGBA")

        # Tensor final
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        # Palette
        colors = [tuple(int(p["color"][i:i + 2], 16) for i in (1, 3, 5)) for p in gradient_points]
        palette_img = Image.new("RGB", (len(colors) * 20, 20))
        draw_pal = ImageDraw.Draw(palette_img)
        for i, color in enumerate(colors):
            draw_pal.rectangle([i * 20, 0, (i + 1) * 20, 20], fill=color)
        palette_tensor = torch.from_numpy(np.array(palette_img).astype(np.float32) / 255.0)[None,]
        palette_hex = ", ".join([p["color"] for p in gradient_points])

        return (img_tensor, palette_tensor, palette_hex)

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#

class ImageBlendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": ([
                    "normal", "multiply", "screen", "overlay",
                    "add", "subtract", "difference", "lighten", "darken"
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "Image/Blend"

    # Définir les couleurs directement dans la classe
    @classmethod  
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    # ------------------------------------------------ #

    def add_alpha(self, arr):
        """Ajoute un canal alpha (opaque) si absent"""
        if arr.shape[2] == 3:
            alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=2)
        return arr

    def resize_and_crop_to_cover(self, arr, target_h, target_w):
        """Resize + crop pour que arr recouvre entièrement (cover) la cible"""
        h, w = arr.shape[:2]

        # Calcul du scale factor (cover)
        scale = max(target_w / w, target_h / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize avec PIL
        img_pil = Image.fromarray((arr * 255).astype(np.uint8)) if arr.dtype != np.uint8 else Image.fromarray(arr)
        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        arr_resized = np.array(img_resized).astype(np.float32) / 255.0

        # Crop centré
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        arr_cropped = arr_resized[start_y:start_y + target_h, start_x:start_x + target_w, :]

        return arr_cropped

    def blend_mode(self, a, b, mode):
        if mode == "normal":
            return b
        elif mode == "multiply":
            return a * b
        elif mode == "screen":
            return 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        elif mode == "add":
            return np.clip(a + b, 0, 1)
        elif mode == "subtract":
            return np.clip(a - b, 0, 1)
        elif mode == "difference":
            return np.abs(a - b)
        elif mode == "lighten":
            return np.maximum(a, b)
        elif mode == "darken":
            return np.minimum(a, b)
        else:
            return b

    def blend(self, image_a, image_b, mode, opacity):
        arr_a = image_a[0].cpu().numpy()
        arr_b = image_b[0].cpu().numpy()

        # Normalisation [0,1]
        if arr_a.max() > 1.0: arr_a = arr_a / 255.0
        if arr_b.max() > 1.0: arr_b = arr_b / 255.0

        # Ajout alpha si manquant
        arr_a = self.add_alpha(arr_a)
        arr_b = self.add_alpha(arr_b)

        # Resize image_b pour couvrir image_a
        arr_b = self.resize_and_crop_to_cover(arr_b, arr_a.shape[0], arr_a.shape[1])

        # Blend par mode
        blended_rgb = self.blend_mode(arr_a[..., :3], arr_b[..., :3], mode)

        # Gestion alpha avec opacity
        alpha_a = arr_a[..., 3:4]
        alpha_b = arr_b[..., 3:4] * opacity
        out_rgb = (1 - alpha_b) * arr_a[..., :3] + alpha_b * blended_rgb

        # Clamp [0,1]
        out_rgb = np.clip(out_rgb, 0, 1)

        # Conversion tensor
        out_tensor = torch.from_numpy(out_rgb).unsqueeze(0).float()

        return (out_tensor,)

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#

class GreatRandomOrganicGradientNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "colors": ("INT", {"default": 3, "min": 2, "max": 8}),
                "blob_count": ("INT", {"default": 5, "min": 1, "max": 50}),
                "blob_shape": (["circle", "square", "polygon", "radial"], {"default": "circle"}),
                "blur_strength": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05}),
                "background_color": ("STRING", {"default": "#FFFFFF", "widget": {"type": "color", "format":"hex"}}),
                "random_background": ("BOOLEAN", {"default": False}),
                "random_palette": ("BOOLEAN", {"default": True}),
                "color1": ("STRING", {"default": "#c7c7c7", "widget": {"type": "color", "format":"hex"}}),
                "color2": ("STRING", {"default": "#4ECDC4", "widget": {"type": "color", "format":"hex"}}),
                "color3": ("STRING", {"default": "#45B7D1", "widget": {"type": "color", "format":"hex"}}),
                "color4": ("STRING", {"default": "#FFA07A", "widget": {"type": "color", "format":"hex"}}),
                "color5": ("STRING", {"default": "#98D8C8", "widget": {"type": "color", "format":"hex"}}),
                "color6": ("STRING", {"default": "#F7DC6F", "widget": {"type": "color", "format":"hex"}}),
                "color7": ("STRING", {"default": "#BB8FCE", "widget": {"type": "color", "format":"hex"}}),
                "color8": ("STRING", {"default": "#85C1E2", "widget": {"type": "color", "format":"hex"}}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 999999}),
                "transparent_background": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "palette_image", "palette_hex")
    FUNCTION = "make_gradient"
    CATEGORY = "Random/Generators"

    def make_gradient(self, width, height, colors, blob_count, blob_shape, blur_strength,
                      background_color, random_background, random_palette,
                      color1, color2, color3, color4, color5, color6, color7, color8,
                      seed, transparent_background):

        # Seed
        if seed == -1:
            seed = np.random.randint(0, 999999)
        rng = np.random.default_rng(seed)

        # Background
        if transparent_background:
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img, "RGBA")
        else:
            if random_background:
                bg = tuple(rng.integers(0, 256, size=3))
            else:
                bg = self._hex_to_rgb(background_color)
            img = Image.new("RGBA", (width, height), (*bg, 255))
            draw = ImageDraw.Draw(img, "RGBA")

        # Palette
        if random_palette:
            palette = [tuple(rng.integers(0, 256, size=3)) for _ in range(colors)]
        else:
            chosen = []
            for c in [color1, color2, color3, color4, color5, color6, color7, color8]:
                if c and c.startswith("#") and len(c) == 7:
                    try:
                        rgb = self._hex_to_rgb(c)
                        chosen.append(rgb)
                    except:
                        pass
            palette = chosen.copy()
            while len(palette) < colors:
                palette.append(tuple(rng.integers(0, 256, size=3)))
            palette = palette[:colors]

        # Dessin des blobs
        for i in range(blob_count):
            color = palette[i % len(palette)]
            x, y = rng.integers(0, width), rng.integers(0, height)
            radius = rng.integers(width // 6, width // 2)

            if blob_shape == "circle":
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(bbox, fill=(*color, 255))

            elif blob_shape == "square":
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.rectangle(bbox, fill=(*color, 255))

            elif blob_shape == "polygon":
                points = [(x + rng.integers(-radius, radius),
                           y + rng.integers(-radius, radius)) for _ in range(rng.integers(3, 8))]
                draw.polygon(points, fill=(*color, 255))

            elif blob_shape == "radial":
                size = radius * 2
                yy, xx = np.mgrid[-radius:radius, -radius:radius]
                dist = np.sqrt(xx**2 + yy**2) / radius
                mask = (1 - np.clip(dist, 0, 1)) * 255
                mask = mask.astype(np.uint8)

                grad = Image.fromarray(mask, mode="L")
                blob = Image.new("RGBA", (size, size), (*color, 255))
                blob.putalpha(grad)
                img.paste(blob, (x - radius, y - radius), blob)

        # Flou
        blur_strength = max(0.05, min(blur_strength, 1.0))
        blur_radius = max(1, int((width + height) / 2 * blur_strength / 4))
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Conversion tensor image principale (toujours RGBA)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.shape[-1] == 3:  # sécurité
            alpha = np.ones((*arr.shape[:2], 1), dtype=np.float32)
            arr = np.concatenate([arr, alpha], axis=-1)
        tensor = torch.from_numpy(arr).unsqueeze(0)

        # Palette en hex
        palette_hex = [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in palette]
        palette_hex_str = ", ".join(palette_hex)

        # Génération image palette (en RGBA aussi)
        palette_img = Image.new("RGBA", (max(256, 64 * colors), 64), (255, 255, 255, 255))
        draw = ImageDraw.Draw(palette_img)
        band_w = palette_img.width // colors
        for i, c in enumerate(palette):
            x0 = i * band_w
            x1 = (i + 1) * band_w if i < colors - 1 else palette_img.width
            draw.rectangle([x0, 0, x1, 64], fill=(*c, 255))

        palette_arr = np.array(palette_img).astype(np.float32) / 255.0
        if palette_arr.shape[-1] == 3:
            alpha = np.ones((*palette_arr.shape[:2], 1), dtype=np.float32)
            palette_arr = np.concatenate([palette_arr, alpha], axis=-1)
        palette_tensor = torch.from_numpy(palette_arr).unsqueeze(0)

        return (tensor, palette_tensor, palette_hex_str)
    
    def _hex_to_rgb(self, hex_color):
        """Convertit une couleur hex en tuple RGB"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#

# ----------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------#



NODE_CLASS_MAPPINGS = {
    "GreatConditioningModifier": GreatConditioningModifier,
    "InteractiveOrganicGradientNode": InteractiveOrganicGradientNode,
    "ImageBlendNode": ImageBlendNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GreatConditioningModifier": "🍄Great Conditioning Modifier",
    "InteractiveOrganicGradientNode": "🍄Great Interactive Organic Gradient",
    "ImageBlendNode": "🍄Great Image Blend"
}