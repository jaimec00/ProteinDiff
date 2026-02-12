
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from proteus.model.base import Base
from proteus.data.data_utils import DataBatch
from proteus.model.model_utils.mlp import SeqProjectionHead, SeqProjectionHeadCfg
from proteus.model.transformer.transformer import TransformerModel, TransformerModelCfg
from proteus.model.tokenizer.wave_func_tokenizer.wf_tokenizer import WaveFunctionTokenizerCfg, WaveFunctionTokenizer
from proteus.utils.masking import Masker, MaskerCfg
from proteus.types import Float, T
    
@dataclass
class proteusAICfg:
    d_model: int
    masker: MaskerCfg = field(default_factory=MaskerCfg)
    tokenizer: WaveFunctionTokenizerCfg = field(default_factory=WaveFunctionTokenizerCfg)
    transformer: TransformerModelCfg = field(default_factory=TransformerModelCfg)
    seq_proj_head: SeqProjectionHeadCfg = field(default_factory=SeqProjectionHeadCfg)
    model_cls: str = "proteus.model.composed.inverse_folding.proteusAI.proteusAI"


class proteusAI(Base):
    def __init__(self, cfg: proteusAICfg):
        super().__init__()

        self.masker = Masker(cfg.masker)
        self.tokenizer = WaveFunctionTokenizer(cfg.tokenizer)
        self.transformer = TransformerModel(cfg.transformer)
        self.seq_proj_head = SeqProjectionHead(cfg.seq_proj_head)

    def forward(self, data_batch: DataBatch) -> Float[T, "BL AA"]:

        aas, is_masked = self.masker.mask_labels(labels=data_batch.labels, also_mask=data_batch.seq_mask)
        data_batch.loss_mask &= is_masked

        wf = self.tokenizer(
            data_batch.coords_ca, 
            data_batch.coords_cb_unit,
            aas,
            data_batch.cu_seqlens,
        )

        latent = self.transformer(
            wf, wf,
            data_batch.cu_seqlens, data_batch.cu_seqlens,
            data_batch.max_seqlen, data_batch.max_seqlen,
        )

        seq_logits = self.seq_proj_head(latent)

        return {
            "seq_logits": seq_logits, 
            "seq_labels": data_batch.labels, 
            "loss_mask": data_batch.loss_mask
        }
