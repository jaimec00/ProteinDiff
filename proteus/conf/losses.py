from proteus.losses.training_loss import LossFnCfg
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore 
from proteus.types import Dict, List, Any

@dataclass
class SeqCelLossFn(LossFnCfg):
    	weights: Dict[str, float] = field(default_factory=lambda: {"seq_cel": 0.0, "seq_focal_loss": 1.0, "seq_probs": 0.0, "seq_acc": 0.0})
    	loss_fns: Dict[str, str] = field(default_factory=lambda: {"seq_cel": "cel", "seq_focal_loss": "focal_loss", "seq_probs": "probs", "seq_acc": "matches"})
    	inputs: Dict[str, List[Any]] = field(default_factory=lambda: {
			"seq_cel": (["seq_logits", "seq_labels", "loss_mask"], {}),
			"seq_focal_loss": (["seq_logits", "seq_labels", "loss_mask"], {"alpha": 1.0, "gamma": 2.0}),
			"seq_probs": (["seq_logits", "seq_labels", "loss_mask"], {}),
			"seq_acc": (["seq_logits", "seq_labels", "loss_mask"], {}),

		})

def register_losses():
  cs = ConfigStore.instance()
  cs.store(name="cel_loss", node=SeqCelLossFn, group="losses")