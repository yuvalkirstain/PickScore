import subprocess
from typing import MutableMapping, Any, Dict

import rich.tree
import rich.syntax
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger(__name__)


def nvidia_smi_gpu_memory_stats():
    """
    Parse the nvidia-smi output and extract the memory used stats.
    """
    out_dict = {}
    try:
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        out_dict = {}
        for item in out_list:
            if " MiB" in item:
                gpu_idx, mem_used = item.split(',')
                gpu_key = f"gpu_{gpu_idx}_mem_used_gb"
                out_dict[gpu_key] = int(mem_used.strip().split(" ")[0]) / 1024
    except FileNotFoundError:
        logger.error(
            "Failed to find the 'nvidia-smi' executable for printing GPU stats"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi returned non zero error code: {e.returncode}")

    return out_dict


def get_nvidia_smi_gpu_memory_stats_str():
    return f"nvidia-smi stats: {nvidia_smi_gpu_memory_stats()}"


def print_config(cfg: DictConfig):
    style = "bright"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    fields = cfg.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)


def _flatten_dict(params: MutableMapping, delimiter: str = "/", parent_key: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            result = {**result, **_flatten_dict(v, parent_key=new_key, delimiter=delimiter)}
        else:
            result[new_key] = v
    return result
