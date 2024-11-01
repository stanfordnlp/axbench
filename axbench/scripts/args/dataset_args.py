from dataclasses import dataclass, field
import argparse
import yaml
import sys
from typing import Optional, List, Any, Dict, Type

@dataclass
class DatasetArgs:
    # Define all parameters with type annotations and optional default values
    models: field(default_factory = lambda: [])
    steering_datasets: field(default_factory = lambda: [])
    input_length: Optional[int] = 32
    output_length: Optional[int] = 10
    data_dir: Optional[str] = None
    train_dir: Optional[str] = None
    dump_dir: Optional[str] = None
    concept_path: Optional[str] = None
    num_of_examples: Optional[int] = None
    latent_num_of_examples: Optional[int] = None
    rotation_freq: Optional[int] = 1_000
    seed: Optional[int] = None
    max_concepts: Optional[int] = None
    model_name: Optional[str] = None
    n_steering_factors: Optional[int] = None
    master_data_dir: Optional[str] = None # this syncs across all jobs.
    steering_batch_size: Optional[int] = None
    steering_output_length: Optional[int] = None
    steering_num_of_examples: Optional[int] = None
    # Add any other parameters as needed

    def __init__(
        self,
        description: str = "Dataset Creation",
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True
    ):
        """
        Initializes DatasetArgs by parsing command-line arguments and loading configurations from a YAML file.
        """
        parser = argparse.ArgumentParser(description=description)

        # Add config file argument
        parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='Path to the YAML configuration file.'
        )

        # Add arguments corresponding to the dataclass fields
        fields = self.__dataclass_fields__
        for field_name, field_def in fields.items():
            # Skip fields that should not be parsed from the command line
            if field_name in ['config_file']:
                continue

            arg_type = self._get_argparse_type(field_def.type)
            parser.add_argument(
                f'--{field_name}',
                type=arg_type,
                help=f'Specify {field_name}.',
            )

        # Add any custom arguments provided
        if custom_args:
            for arg in custom_args:
                parser.add_argument(*arg['args'], **arg['kwargs'])

        args = parser.parse_args()

        # Load the YAML configuration file
        config_file_path = args.config
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # Initialize attributes from config
        for field_name in fields:
            if field_name == 'config_file':
                continue
            value = config.get(field_name, None)
            setattr(self, field_name, value)

        # Overwrite with command-line arguments if provided
        if override_config:
            for field_name in vars(args):
                if field_name in ['config']:
                    continue  # Skip the config file argument itself
                arg_value = getattr(args, field_name)
                if arg_value is not None:
                    setattr(self, field_name, arg_value)

        # Additional attributes
        self.config_file = config_file_path

        # Print the final configuration
        print("Final Configuration:")
        for key in fields:
            print(f"{key}: {getattr(self, key)}")

    @staticmethod
    def _get_argparse_type(field_type: Type) -> Type:
        """
        Helper method to get the argparse type from the dataclass field type.
        """
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
