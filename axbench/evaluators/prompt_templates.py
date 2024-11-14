UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence. 

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated, and point out references to the concept in the fragment. \
After providing your explanation, rate 1 if the concept is incorporated and rate 0 otherwise \
by following this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate 1 if the sentence is \
related to the instruction and rate 0 otherwise by following this exact format: \
"Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""