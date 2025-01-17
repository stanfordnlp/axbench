## Integration with AxBench
This document will guide you how to integrate your methods with AxBench. You can also add new evaluators to AxBench. We try to implement this codebase such as that you can easily add new methods and evaluators.

### How to add new methods?
All of our existing methods are implemented in `axbench/models`. You can add your methods by implementing the `BaseModel` interface. Mainly, you need to implement these methods:

```python
class BaseModel(object):
    """Base class for all models."""
    def __init__(self, **kwargs):
        """Initialize your model. See other models for kwargs annotations."""
        pass

    def __str__(self):
        """Return the name of your model."""
        pass

    def make_model(self, **kwargs):
        """Make your model (i.e., how to initialize your models from scratch)."""
        pass

    def make_dataloader(self, examples, **kwargs):
        """Make your dataloader."""
        pass

    def train(self, examples, **kwargs):
        """Train your model."""
        pass

    def save(self, dump_dir, **kwargs):
        """Save your model."""
        pass

    def load(self, dump_dir, **kwargs):
        """Load your model."""
        pass

    def predict_latent(self, examples, **kwargs):
        """Concept detection. This function should return the predicted latent activations of given examples."""
        pass    

    def predict_steer(self, examples, **kwargs):
        """Model steering. This function should return steerted results."""
        pass

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        """This function should pre-compute mean activations for your model. The activations will be used for model steering."""
``` 

### How to add new evaluators?
You can add new evaluators by implementing the `BaseEvaluator` interface. Mainly, you need to implement these methods:
```python
class Evaluator(ABC):

    def __str__(self):
        """Return the name of your evaluator."""
        pass

    def fit(self, examples):
        """
        This is a placeholder in case then evaluator
        actually needs to be trained.
        """
        pass

    @abstractmethod
    def compute_metrics(self, examples):
        """Compute metrics. The returned metrics will be saved in dict here the key is the name of your evaluator."""
        pass
```

### How to run your new method and evaluator?
After you implement your new method and evaluator, you can create new yamls or modify existing yamls to run your new method and evaluator. Our existing yamls are in `axbench/sweep`. Existing yamls are already built per method and evaluator. You can modify them to run your new method and evaluator.

