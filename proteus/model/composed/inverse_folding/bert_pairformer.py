import torch
import torch.nn as nn
from omegaconf import DictConfig
from dataclasses import dataclass, field
from proteus.static.constants import canonical_aas, alphabet

from proteus.model.base import Base
from proteus.data.data_utils import DataBatch
from proteus.model.transformer.transformer import PairformerModelCfg, TransformerModelCfg, TransformerModel
from proteus.model.transformer.moq import PairAggregator, PairAggregatorCfg
from proteus.model.model_utils.mlp import SeqProjectionHead, SeqProjectionHeadCfg
from proteus.model.tokenizer.structure_tokenizer import StructureTokenizer, StructureTokenizerCfg

@dataclass
class BertPairformerCfg:
    d_model: int
    d_pair: int
    tokenizer: StructureTokenizerCfg = field(default_factory=StructureTokenizerCfg)
    pairformer: PairformerModelCfg = field(default_factory=PairformerModelCfg)
    pair_aggregator: PairAggregatorCfg = field(default_factory=PairAggregatorCfg)
    transformer: TransformerModelCfg = field(default_factory=TransformerModelCfg)
    seq_proj_head: SeqProjectionHeadCfg = field(default_factory=SeqProjectionHeadCfg)
    model_cls: str = "proteus.model.composed.inverse_folding.bert_pairformer.BertPairformer"

class BertPairformer(Base):
    def __init__(self, cfg: BertPairformerCfg):
        super().__init__()

        self.tokenizer = StructureTokenizer(cfg.tokenizer)
        self.pairformer = TransformerModel(cfg.pairformer)
        self.pair_aggregator = PairAggregator(cfg.pair_aggregator)
        self.transformer = TransformerModel(cfg.transformer)
        self.seq_proj = SeqProjectionHead(cfg.seq_proj_head)

    def forward(self, data_batch: DataBatch):

        struct_embeddings = self.tokenizer(
            data_batch.coords_bb_dist,
            data_batch.rel_frames,
            data_batch.rel_seq_idx, 
            data_batch.diff_chain, 
        )
        struct_logits = self.pairformer(struct_embeddings, data_batch.pair_cu_seqlens, data_batch.max_seqlen)
        seq_logits = torch.utils.checkpoint.checkpoint(self.pair_aggregator, struct_logits, data_batch.pair_reduce_idxs, data_batch.reduction_buffer, use_reentrant=False)
        seq_logits = self.transformer(seq_logits, data_batch.cu_seqlens, data_batch.max_seqlen)
        seq_logits = self.seq_proj(seq_logits)

        return {"seq_logits": seq_logits, "seq_labels": data_batch.labels, "loss_mask": data_batch.loss_mask}