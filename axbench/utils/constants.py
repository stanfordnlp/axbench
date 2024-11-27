#################################
#
# Constants.
#
#################################


from enum import Enum

class EXAMPLE_TAG(Enum):
    CONTROL = 0
    EXPERIMENT = 1

OPENAI_RATE_LIMIT = 10
PRICING_DOLLAR_PER_1M_TOKEN = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

UNIT_1M = 1_000_000

CHAT_MODELS = {
    "google/gemma-2-2b-it"
}

BASE_MODELS = {
    "google/gemma-2-2b"
}

EMPTY_CONCEPT = "EEEEE"