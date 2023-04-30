from dataclasses import dataclass
from accelerate.utils import DummyOptim


@dataclass
class DummyOptimizerConfig:
    _target_: str = "trainer.optimizers.dummy_optimizer.BaseDummyOptim"
    lr: float = 3e-6
    weight_decay: float = 0.3


class BaseDummyOptim(DummyOptim):
    def __init__(self, model, lr=0.001, weight_decay=0, **kwargs):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs
