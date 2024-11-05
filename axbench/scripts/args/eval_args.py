from dataclasses import dataclass, field
import argparse
import yaml
from typing import Optional, List, Type

@dataclass
class EvalArgs:
    models: field(default_factory=lambda: [])
    latent_evaluators: field(default_factory=lambda: [])
    steering_evaluators: field(default_factory=lambda: [])
    report_to: field(default_factory=lambda: [])
    rotation_freq: Optional[int] = 1_000
    data_dir: Optional[str] = None
    dump_dir: Optional[str] = None
    num_of_workers: Optional[int] = 16
    lm_model: Optional[str] = None
    run_winrate: Optional[bool] = None
    winrate_baseline: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    run_name: Optional[bool] = None

    def __init__(
        self,
        description: str = "Evaluation Script",
        config_file: str = None,
        section: str = "train",  # Specify section to load
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True
    ):
        """
        Initializes EvalArgs by parsing command-line arguments and loading configurations from a YAML file.
        """
        parser = argparse.ArgumentParser(description=description)

        # Add config file argument
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        # Add arguments corresponding to the dataclass fields
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

        # Add any custom arguments provided
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
