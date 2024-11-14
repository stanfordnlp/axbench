UNIDIRECTIONAL_RATING_NO_ICL_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the response fragment below. The response should \
have words or phrases related to both the concept and the instruction even if it is not coherent. \
Failure to do so should receive the lowest score. If the response has words or phrases related to both \
the concept and the instruction, it should receive a score based on its language quality and coherence. 

Provide a detailed explanation addressing these points. Then, rate the response on a scale from **1** \
to **10**, where a higher score reflects better incorporation of the concept and adherence to the instruction. \
Use the format: "Rating: [[score]]".

[Instruction Start]
%s
[Instruction End]

[Concept Start]
%s
[Concept End]

[Response Fragment Start]
%s
[Response Fragment End]
"""


UNIDIRECTIONAL_RATING_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the response fragment below. The response should \
have words or phrases related to both the concept and the instruction even if it is not coherent. \
Failure to do so should receive the lowest score. If the response has words or phrases related to both \
the concept and the instruction, it should receive a score based on its language quality and coherence. 

Provide a detailed explanation addressing these points. Then, rate the response on a scale from **1** \
to **10**, where a higher score reflects better incorporation of the concept and adherence to the instruction. \
Use the format: "Rating: [[score]]".

**Demonstration Examples Start**

[Instruction Start]
How can I improve my graphic design skills?
[Instruction End]

[Concept Start]
Scientific terminology and concepts related to biological processes and medical conditions
[Concept End]

[Response Fragment Start]
The quick brown fox jumps over the lazy dog multiple times in the evening. Colors and shapes make pictures look nice, \
and sometimes the weather is warm when people are outside. Walking through the park, the leaves rustle underfoot, and \
the sky is blue with white clouds floating by. Birds sing songs that are pleasant to hear while you think about various things unrelated to design.
[Response Fragment End]

The response does not address the instruction about improving graphic design skills and fails to mention or incorporate the \
specified concept of scientific terminology related to biological processes and medical conditions. The content is random and \
unrelated, consisting of general statements about nature and activities. Since it does not meet the hard requirements of \
concept inclusion and instruction relevance, it receives the lowest score.

Rating: [[1]]


[Instruction Start]
Make a list of snacks and foods to serve as party snacks on a game day!
[Instruction End]

[Concept Start]
Java programming code involving data structures and method calls.
[Concept End]

[Response Fragment Start]
## Game Day Snacks: Supercharge Your Party! 

**Classic Favorites:**

* **Chips & Dips:**
    * Classic Potato Chips (tortilla, potato, vegetable)
    <br>  <div class="callout<>();ArrayList<String> ArrayList<>();ArrayList<>();ArrayList<>();ArrayList<>();ArrayList<>();\
ArrayList<>();ArrayList<>();ArrayList<>();
[Response Fragment End]

The response begins by addressing the instruction to make a list of snacks for a game day party, \
mentioning "Game Day Snacks" and listing "Chips & Dips" and "Classic Potato Chips." However, \
it does not provide a comprehensive or coherent list of snacks. It attempts to incorporate the concept \
by including Java programming code involving data structures and method calls, evident from the repeated \
ArrayList<String> ArrayList<>(); statements. However, this code is randomly inserted and disrupts the flow, \
making the response confusing. The language quality is poor due to the incoherent mix of snack listings and \
fragmented Java code, which hinders understanding. Since the response mentions both the instruction and the concept, \
albeit poorly executed, it should receive a score of 2.

Rating: [[2]]


[Instruction Start]
How can I improve my public speaking skills?
[Instruction End]

[Concept Start]
Python code related to date and time functionalities
[Concept End]

[Response Fragment Start]
Improving your public speaking skills is like refining Python code for date and time functionalities. You need to plan ahead, \
much like scheduling events with the datetime module. Practice regularly to stay in sync, similar to how cron jobs execute code \
at scheduled intervals. By debugging your 'syntax' and receiving feedback, you can eliminate errors in your delivery.
[Response Fragment End]

The response addresses the instruction by offering advice on improving public speaking skills. It meaningfully \
incorporates the concept by drawing parallels with Python's date and time functionalities, mentioning the `datetime` module \
and cron jobs. The language is coherent and understandable.

Rating: [[10]]

**Demonstration Examples End**


[Instruction Start]
%s
[Instruction End]

[Concept Start]
%s
[Concept End]

[Response Fragment Start]
%s
[Response Fragment End]
"""


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