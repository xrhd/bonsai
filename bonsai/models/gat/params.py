from bonsai.models.gat import modeling


def _get_key_and_transform_mapping(cfg: modeling.GATConfig):
    # Placeholder for mapping GAT weights if we were to load them from a safetensors file
    # For now, we return an empty dict or a hypothetical mapping
    return {}


def load_pretrained_gat(file_path: str, config: modeling.GATConfig):
    raise NotImplementedError("Pretrained weights loading for GAT is not yet implemented.")
