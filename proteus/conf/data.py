from proteus.data.data_loader import DataHolderCfg
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class DefaultData(DataHolderCfg):
    data_path: str = "/home/ubuntu/proteinDiffVirgina/data/pdb_2021aug02"
    num_train: int = -1
    num_val: int = -1
    num_test: int = -1
    batch_tokens: int = 512
    min_seq_size: int = 16
    max_seq_size: int = 512
    max_resolution: float = 3.5
    homo_thresh: float = 0.7
    asymmetric_units_only: bool = False
    num_workers: int = 16
    prefetch_factor: int = 8
    rng_seed: int = 42
    buffer_size: int = 64

@dataclass
class XSmallSeqData(DefaultData):
    batch_tokens: int = 128
    max_seq_size: int = 128

@dataclass
class SmallSeqData(DefaultData):
    batch_tokens: int = 512
    max_seq_size: int = 512

@dataclass
class MediumSeqData(DefaultData):
    batch_tokens: int = 1024
    max_seq_size: int = 1024

@dataclass
class LargeSeqData(DefaultData):
    batch_tokens: int = 8192
    max_seq_size: int = 8192

@dataclass
class XLargeSeqData(DefaultData):
    batch_tokens: int = 16384
    max_seq_size: int = 16384

@dataclass
class SmallData(DefaultData):
    num_train: int = 128
    num_val: int = 128
    num_test: int = 128

def register_data():
    cs = ConfigStore.instance()
    cs.store(name="default", node=DefaultData, group="data")
    cs.store(name="extra_small_seq", node=SmallSeqData, group="data")
    cs.store(name="small_seq", node=SmallSeqData, group="data")
    cs.store(name="medium_seq", node=SmallSeqData, group="data")
    cs.store(name="large_seq", node=SmallSeqData, group="data")
    cs.store(name="extra_large_seq", node=SmallSeqData, group="data")
    cs.store(name="small", node=SmallData, group="data")