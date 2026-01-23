# ----------------------------------------------------------------------------------------------------------------------
'''
author:         jaime cardenas
title:          eval_vae.py
description:    Evaluates trained VAE model by encoding test data and sampling from latent distribution
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import hydra
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf as om, DictConfig
from typing import Optional, List

from proteindiff.model import ProteinDiff
from proteindiff.model.ProteinDiff import ProteinDiffCfg
from proteindiff.training.data.data_loader import DataHolder, DataHolderCfg
from proteindiff.static.constants import TrainingStage
from proteindiff.utils.generation_utils import save_pdb
from proteindiff.utils.struct_utils import compute_rmsd, coords_from_txy_sincos
from proteindiff.static import residue_constants as rc

# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class EvalVAECfg:
    checkpoint_path: str
    output_dir: str
    num_samples: int = 10  # Number of latent samples per input
    data: DataHolderCfg = None

class EvalVAE:
    def __init__(self, cfg: EvalVAECfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint and model
        self.model = self.load_checkpoint(cfg.checkpoint_path)
        self.model.eval()

        # Load data
        self.data_holder = DataHolder(cfg.data)

        print(f"Loaded checkpoint from {cfg.checkpoint_path}")
        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sampling {cfg.num_samples} structures per input")

    def load_checkpoint(self, checkpoint_path: str) -> ProteinDiff:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load configs from checkpoint directory
        checkpoint_dir = checkpoint_path.parent
        model_cfg_path = checkpoint_dir / "model_cfg.yaml"

        if not model_cfg_path.exists():
            raise FileNotFoundError(
                f"Model config not found at {model_cfg_path}. "
                "Make sure the checkpoint was saved with the full training config."
            )

        # Load model config
        model_cfg = om.load(model_cfg_path)

        # Create model
        model = ProteinDiff(model_cfg)

        # Load weights
        print(f"Loading weights from {checkpoint_path}")
        weights = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

        if "model" in weights:
            state_dict = weights["model"]
        else:
            state_dict = weights

        # Load state dict
        model.load_state_dict(state_dict, strict=False)

        # Move to device
        model = model.to(self.device)

        return model

    @torch.no_grad()
    def encode(self, data_batch):
        """Run encoder to get mu and logvar."""
        # Move batch to device
        data_batch.move_to(self.device)

        # Run tokenizer
        coords_bb, divergence, frames = self.model.tokenizer(
            data_batch.coords,
            data_batch.labels,
            data_batch.atom_mask
        )

        # Run encoder
        latent, mu, logvar = self.model.vae.encoder(
            divergence=divergence,
            bb_coords=coords_bb,
            frames=frames,
            seq_idx=data_batch.seq_idx,
            chain_idx=data_batch.chain_idx,
            sample_idx=data_batch.sample_idx,
            cu_seqlens=data_batch.cu_seqlens,
            max_seqlen=data_batch.max_seqlen,
        )

        return mu, logvar, data_batch

    @torch.no_grad()
    def sample_latents(self, mu, logvar, num_samples):
        """Sample multiple latents from the latent distribution."""
        # mu, logvar: (ZN, d_latent)
        # We want to sample num_samples latents for each residue

        latents = []
        for _ in range(num_samples):
            # Sample from N(mu, exp(logvar))
            eps = torch.randn_like(mu)
            latent = mu + eps * torch.exp(0.5 * logvar)
            latents.append(latent)

        return latents

    @torch.no_grad()
    def decode(self, latent, cu_seqlens, max_seqlen):
        """Run decoder to get structure predictions."""
        # Run decoder
        divergence_pred, seq_pred, struct_logits, struct_head = self.model.vae.decoder(
            x=latent,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Get predicted sequence (argmax)
        seq_pred_labels = seq_pred.argmax(dim=-1)

        # Get structure predictions from struct_head
        txy = struct_head.frame_proj(struct_logits)
        sincos = struct_head.torsion_proj(struct_logits)

        t, x, y = torch.chunk(txy, chunks=3, dim=-1)
        sin, cos = torch.chunk(sincos, chunks=2, dim=-1)

        # Reconstruct coordinates using predicted sequence
        coords_pred, atom_mask_pred = coords_from_txy_sincos(
            t, x, y, sin, cos, seq_pred_labels,
            struct_head.restype_rigid_group_default_frame,
            struct_head.restype_atom14_rigid_group_positions,
            struct_head.restype_atom14_to_rigid_group,
            struct_head.restype_atom14_mask,
            struct_head.chi_angles_mask
        )

        return coords_pred, seq_pred_labels, atom_mask_pred

    def split_batch(self, coords, labels, atom_mask, cu_seqlens):
        """Split batch tensors into individual samples using cu_seqlens."""
        num_samples = len(cu_seqlens) - 1
        samples = []

        for i in range(num_samples):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()

            sample = {
                'coords': coords[start_idx:end_idx],
                'labels': labels[start_idx:end_idx],
                'atom_mask': atom_mask[start_idx:end_idx]
            }
            samples.append(sample)

        return samples

    def evaluate(self):
        """Evaluate VAE on test set."""
        print("\nStarting VAE evaluation on test set...")

        test_loader = self.data_holder.test
        total_batches = len(test_loader)

        all_rmsds = []
        structure_count = 0

        with tqdm(total=total_batches, desc="Evaluating") as pbar:
            for batch_idx, data_batch in enumerate(test_loader):

                # Encode to get mu and logvar
                mu, logvar, data_batch = self.encode(data_batch)

                # Sample multiple latents
                latents = self.sample_latents(mu, logvar, self.cfg.num_samples)

                # For each latent sample
                for sample_idx, latent in enumerate(latents):

                    # Decode to get coordinates (batched)
                    coords_pred, seq_pred, atom_mask_pred = self.decode(
                        latent,
                        data_batch.cu_seqlens,
                        data_batch.max_seqlen,
                    )

                    # Split batch into individual samples
                    pred_samples = self.split_batch(
                        coords_pred,
                        seq_pred,
                        atom_mask_pred,
                        data_batch.cu_seqlens
                    )

                    true_samples = self.split_batch(
                        data_batch.coords,
                        data_batch.labels,
                        data_batch.atom_mask,
                        data_batch.cu_seqlens
                    )

                    # Save each sample in the batch
                    for inner_idx, (pred_sample, true_sample) in enumerate(zip(pred_samples, true_samples)):

                        # Get backbone mask (N, CA, C, CB = atoms 0, 1, 2, 3)
                        backbone_mask = torch.zeros_like(true_sample['atom_mask'])
                        backbone_mask[:, :4] = true_sample['atom_mask'][:, :4]

                        # Compute RMSD (backbone only)
                        rmsd = compute_rmsd(
                            pred_sample['coords'],
                            true_sample['coords'],
                            backbone_mask
                        )
                        all_rmsds.append(rmsd)

                        # Save PDB file
                        output_dir = self.output_dir / f"batch{batch_idx:04d}" / f"inner{inner_idx:02d}" / f"sample{sample_idx:02d}"
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Use predicted sequence
                        save_pdb(
                            coords=pred_sample['coords'],
                            labels=pred_sample['labels'],
                            output_path=output_dir / "pred.pdb",
                            atom_mask=pred_sample['atom_mask'],
                            chain_id="A"
                        )

                        # save the true sample to compare
                        save_pdb(
                                coords=true_sample['coords'],
                                labels=true_sample['labels'],
                                output_path=output_dir / "true.pdb",
                                atom_mask=true_sample['atom_mask'],
                                chain_id="A"
                            )

                        structure_count += 1

                # Update progress bar with average RMSD
                if all_rmsds:
                    avg_rmsd = sum(all_rmsds) / len(all_rmsds)
                    pbar.set_postfix({"avg_rmsd": f"{avg_rmsd:.3f} Å"})

                pbar.update(1)

        # Print summary statistics
        print(f"\nEvaluation complete!")
        print(f"Total structures generated: {structure_count}")
        print(f"Average backbone RMSD: {sum(all_rmsds) / len(all_rmsds):.3f} Å")
        print(f"Min RMSD: {min(all_rmsds):.3f} Å")
        print(f"Max RMSD: {max(all_rmsds):.3f} Å")
        print(f"Output saved to: {self.output_dir}")

        # Save RMSD statistics to file
        stats_file = self.output_dir / "rmsd_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Total structures: {structure_count}\n")
            f.write(f"Average RMSD: {sum(all_rmsds) / len(all_rmsds):.3f} Å\n")
            f.write(f"Min RMSD: {min(all_rmsds):.3f} Å\n")
            f.write(f"Max RMSD: {max(all_rmsds):.3f} Å\n")
            f.write(f"\nAll RMSDs:\n")
            for i, rmsd in enumerate(all_rmsds):
                f.write(f"{i}: {rmsd:.3f} Å\n")

@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):

    # Create eval config
    eval_cfg = EvalVAECfg(
        checkpoint_path=cfg.checkpoint_path,
        output_dir=cfg.output_dir,
        num_samples=cfg.num_samples,
        data=cfg.data,
    )

    # Run evaluation
    evaluator = EvalVAE(eval_cfg)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
