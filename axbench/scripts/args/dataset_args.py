from dataclasses import dataclass, field
import argparse
import yaml
from typing import Optional, List, Type

@dataclass
class DatasetArgs:
    models: field(default_factory=lambda: [])
    steering_datasets: field(default_factory=lambda: [])
    input_length: Optional[int] = 32
    output_length: Optional[int] = 32
    inference_batch_size: Optional[int] = 8
    # generation related params
    temperature: Optional[float] = 1.0
    data_dir: Optional[str] = None
    train_dir: Optional[str] = None
    dump_dir: Optional[str] = None
    concept_path: Optional[str] = None
    num_of_examples: Optional[int] = None
    latent_num_of_examples: Optional[int] = None
    latent_batch_size: Optional[int] = None
    rotation_freq: Optional[int] = 1_000
    seed: Optional[int] = None
    max_concepts: Optional[int] = None
    model_name: Optional[str] = None
    steering_model_name: Optional[str] = None
    n_steering_factors: Optional[int] = None
    steering_factors: Optional[List[str]] = None
    steering_layers: Optional[List[int]] = None
    master_data_dir: Optional[str] = None
    steering_batch_size: Optional[int] = None
    steering_output_length: Optional[int] = None
    steering_num_of_examples: Optional[int] = None
    steering_intervention_type: Optional[str] = None
    lm_model: Optional[str] = None
    run_name: Optional[str] = None
    use_bf16: Optional[bool] = False
    dataset_category: Optional[str] = "instruction"
    lm_use_cache: Optional[bool] = True
    disable_neuronpedia_max_act: Optional[bool] = False
    imbalance_factor: Optional[int] = 100
    overwrite_data_dir: Optional[str] = None

    def __init__(
        self,
        description: str = "Dataset Creation",
        config_file: str = None,
        section: str = "train",  # Default to 'train' section
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True
    ):
        parser = argparse.ArgumentParser(description=description)
        
        # Command-line argument for YAML configuration file
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        fields = self.__dataclass_fields__
        for field_name, field_def in fields.items():
            if field_name == 'config_file':
                continue
            arg_type = self._get_argparse_type(field_def.type)
            parser.add_argument(
                f'--{field_name}',
                type=arg_type,
                help=f'Specify {field_name}.',
            )

        if custom_args:
            for arg in custom_args:
                parser.add_argument(*arg['args'], **arg['kwargs'])

        args = parser.parse_args()

        # Load the YAML configuration file
        config_file_path = args.config
        if not config_file_path:
            raise ValueError("A config file must be provided.")
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # Select the specified section
        section_data = config.get(section, {})
        if not section_data:
            raise ValueError(f"Section '{section}' not found in the YAML configuration.")

        # Initialize attributes from the selected section
        for field_name in fields:
            if field_name == 'config_file':
                continue
            value = section_data.get(field_name, None)
            setattr(self, field_name, value)

        # Overwrite with command-line arguments if provided
        if override_config:
            for field_name in vars(args):
                if field_name in ['config']:
                    continue
                arg_value = getattr(args, field_name)
                if arg_value is not None:
                    setattr(self, field_name, arg_value)

        self.config_file = config_file_path

        print("Final Configuration:")
        for key in fields:
            print(f"{key}: {getattr(self, key)}")

    @staticmethod
    def _get_argparse_type(field_type: Type) -> Type:
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
            field_type = field_type.__args__[0]
        if field_type == int:
            return int
        elif field_type == float:
            return float
        elif field_type == bool:
            return lambda x: (str(x).lower() in ['true', '1', 'yes'])
        else:
            return str
