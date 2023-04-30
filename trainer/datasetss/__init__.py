from hydra.core.config_store import ConfigStore

from trainer.datasetss.clip_hf_dataset import CLIPHFDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="clip", node=CLIPHFDatasetConfig)
