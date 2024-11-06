from dataclasses import dataclass
import argparse
import yaml
from typing import Optional, List

class ModelContainer:
    def __init__(self):
        self._models = {}
    
    def add_model(self, name, params):
        self._models[name] = params
        if name.isidentifier():
            setattr(self, name, params)
        else:
            print(f"Warning: Model name '{name}' is not a valid Python identifier. Use dictionary access.")
    
    def __getitem__(self, key):
        return self._models[key]
    
    def __iter__(self):
        return iter(self._models.items())
    
    def keys(self):
        return self._models.keys()
    
    def values(self):
        return self._models.values()
    
    def items(self):
        return self._models.items()

@dataclass
class ModelParams:
    batch_size: Optional[int] = None
    n_epochs: Optional[int] = None
    k_latent_null_loss: Optional[int] = None
    lr: Optional[float] = None
    coeff_l1_loss_null: Optional[float] = None
    coeff_l1_loss: Optional[float] = None

class TrainingArgs:
    def __init__(
        self,
        description: str = "Training Script",
        config_file: str = None,
        section: str = "train",  # Specify section to load
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True
    ):
        parser = argparse.ArgumentParser(description=description)

        # Add config file argument
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        # Define global and hierarchical parameters
        global_params = [
            'concept_path', 'model_name', 'layer', 'component',
            'data_dir', 'dump_dir', 'run_name'
        ]
        hierarchical_params = [
            'batch_size', 'n_epochs', 'k_latent_null_loss',
            'lr', 'coeff_l1_loss_null', 'coeff_l1_loss'
        ]
        all_params = global_params + hierarchical_params

        # Add arguments for all parameters
        for param in all_params:
            parser.add_argument(f'--{param}', type=self._infer_type(param), help=f'Specify {param}.')

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

        # Initialize global parameters
        for param in global_params:
            arg_value = getattr(args, param, None)
            config_value = section_data.get(param, None)
            setattr(self, param, arg_value if arg_value is not None else config_value)

        # Initialize hierarchical parameters with global defaults
        for param in hierarchical_params:
            arg_value = getattr(args, param, None)
            config_value = section_data.get(param, None)
            setattr(self, param, arg_value if arg_value is not None else config_value)

        # Initialize models list
        self.models_list = []
        self.model_params = {}
        if 'models' in section_data:
            if isinstance(section_data['models'], dict):
                self.models_list = list(section_data['models'].keys())
                self.model_params = section_data['models']
            elif isinstance(section_data['models'], list):
                self.models_list = section_data['models']
            else:
                raise ValueError("Invalid format for 'models' in config")
        else:
            self.models_list = []

        # Create models container
        self.models = ModelContainer()

        # Initialize per-model parameters
        for model_name in self.models_list:
            params = ModelParams()
            # Set hierarchical parameters to global defaults
            for param in hierarchical_params:
                setattr(params, param, getattr(self, param, None))
            # Override with per-model parameters if available
            if model_name in self.model_params:
                model_config = self.model_params[model_name]
                for param_name, param_value in model_config.items():
                    if param_name in hierarchical_params:
                        setattr(params, param_name, param_value)
            # Add the model to the container
            self.models.add_model(model_name, params)

        # Additional attributes
        self.config_file = config_file_path

        # Print the final configuration
        print("Final Configuration:")
        print("Global Parameters:")
        for key in global_params + hierarchical_params:
            print(f"{key}: {getattr(self, key)}")
        print("\nPer-Model Parameters:")
        for model_name, params in self.models:
            print(f"{model_name}:")
            for field_name in ModelParams.__dataclass_fields__:
                print(f"  {field_name}: {getattr(params, field_name)}")

    @staticmethod
    def _infer_type(param_name: str):
        int_params = ['layer', 'batch_size', 'n_epochs', 'k_latent_null_loss']
        float_params = ['lr', 'coeff_l1_loss_null', 'coeff_l1_loss']
        str_params = ['concept_path', 'model_name', 'component', 'data_dir', 'dump_dir', 'run_name']

        if param_name in int_params:
            return int
        elif param_name in float_params:
            return float
        elif param_name in str_params:
            return str
        else:
            return str

