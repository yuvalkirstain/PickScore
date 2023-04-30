from dataclasses import dataclass, field
from typing import List, Any, Dict

from omegaconf import DictConfig, MISSING

import trainer.accelerators
import trainer.tasks
import trainer.models
import trainer.criterions
import trainer.datasetss
import trainer.optimizers
import trainer.lr_schedulers
from trainer.accelerators.base_accelerator import BaseAcceleratorConfig
from trainer.models.base_model import BaseModelConfig
from trainer.tasks.base_task import BaseTaskConfig


def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def instantiate_with_cfg(cfg: DictConfig, **kwargs):
    target = _locate(cfg._target_)
    return target(cfg, **kwargs)


defaults = [
    {"accelerator": "deepspeed"},
    {"task": "clip"},
    {"model": "clip"},
    {"criterion": "clip"},
    {"dataset": "clip"},
    {"optimizer": "dummy"},
    {"lr_scheduler": "dummy"},
]


@dataclass
class DebugConfig:
    activate: bool = False
    port: int = 5900


@dataclass
class TrainerConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    accelerator: BaseAcceleratorConfig = MISSING
    task: BaseTaskConfig = MISSING
    model: BaseModelConfig = MISSING
    criterion: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    lr_scheduler: Any = MISSING
    debug: DebugConfig = DebugConfig()
    output_dir: str = "outputs"
