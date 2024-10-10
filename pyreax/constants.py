#################################
#
# Constants.
#
#################################


from enum import Enum

class EXAMPLE_TAG(Enum):
    CONTROL = 0
    EXPERIMENT = 1

OPENAI_RATE_LIMIT = 100
PRICING_DOLLAR_PER_1M_TOKEN = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

UNIT_1M = 1_000_000

TEXT_GENRES = [
    "fictional narrative",
    "non-fictional writing",
    "scientific paper",
    "poetry",
    "journalistic article",
    "technical documentation",
    "biography",
    "essay",
    "script or screenplay",
]

CODE_GENRES = [
    "Python programming code",
    "Java programming code",
    "JavaScript programming code",
    "C++ programming code",
    "C# programming code",
    "Ruby programming code"
]

MATH_GENRES = [
    "algebraic expressions",
    "mathematical equations",
    "logic and set theory",
]