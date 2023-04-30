from hydra.core.config_store import ConfigStore

from trainer.lr_schedulers.constant_with_warmup import ConstantWithWarmupLRSchedulerConfig
from trainer.lr_schedulers.dummy_lr_scheduler import DummyLRSchedulerConfig

cs = ConfigStore.instance()
cs.store(group="lr_scheduler", name="dummy", node=DummyLRSchedulerConfig)
cs.store(group="lr_scheduler", name="constant_with_warmup", node=ConstantWithWarmupLRSchedulerConfig)
