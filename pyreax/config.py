import json
from dataclasses import dataclass, asdict, fields


class ConfigMismatchError(Exception):
    """Custom exception raised when two Config objects do not match."""
    def __init__(self, message):
        super().__init__(message)


@dataclass
class Config:
    concept_path: str | None = None
    
    model_name: str | None = None
    lm_model: str | None = None
    n_data: int = 66
    layer: int = 20
    component: str = "res"

    n_epochs: int = 8
    k_latent_null_loss: int = 3

    coeff_l1_loss_null: float = 5E-2
    coeff_l1_loss: float = 1E-3

    dump_dir: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    def get_model_name(self):
        shorten_model_name = self.model_name.split("/")[-1]
        return f"{shorten_model_name}/{self.layer}-reax-{self.component}-{self.lm_model}"

    def __repr__(self):
        # Convert to dictionary and serialize to JSON-friendly format
        return json.dumps(asdict(self), indent=4)


def load_config_from_json(file_path: str) -> Config:
    # Read the JSON file and convert it to a dictionary
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create a Config object using the dictionary
    return Config(**config_dict)


def compare_configs(config1: Config, config2: Config):
    """Compare two Config objects and raise an exception if they are not equal."""
    unequal_fields = []

    for field in fields(Config):  # Get all the fields of the Config dataclass
        field_name = field.name
        value1 = getattr(config1, field_name)
        value2 = getattr(config2, field_name)

        if value1 != value2:
            unequal_fields.append((field_name, value1, value2))

    if unequal_fields:
        mismatch_details = "\n".join(
            [f"Field '{field}': memory_config={v1}, loaded_config={v2}"
             for field, v1, v2 in unequal_fields]
        )
        raise ConfigMismatchError(
            f"Config objects do not match in the following fields:\n{mismatch_details}"
        )