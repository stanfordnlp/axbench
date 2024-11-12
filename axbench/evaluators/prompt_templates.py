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


UNIDIRECTIONAL_PAIRWISE_EVALUATION_NO_ICL_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the quality of the response \
fragments provided by two AI assistants in answer to the instruction below. \
Your evaluation should prioritize whether each response contains clear \
references or terms related to the specified concept. If only one response \
includes the concept and is grammatically correct, coherent, and aligned \
with the instruction, select that response as the better one. If only one \
response includes the concept but lacks grammatical correctness, coherence, \
or relevance to the instruction, declare a tie. If both responses include the \
concept, assess which response is more relevant to the instruction. \
If both responses include the concept and are equally relevant, coherent, \
and meet all requirements, declare a tie.

Begin by comparing the two response fragments and provide a short explanation \
of your reasoning, highlighting references to the concept as well as terms related \
to the instruction. Avoid any biases related to position, response length, \
assistant names, or response order. Be as objective as possible. After your \
explanation, output your final decision in this exact format: "[[A]]" if \
assistant A is better, "[[B]]" if assistant B is better, or "[[C]]" for a tie.

[Instruction Start]
{instruction}
[Instruction End]

[Concept Start]
{concept}
[Concept End]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""


UNIDIRECTIONAL_PAIRWISE_EVALUATION_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the quality of the response \
fragments provided by two AI assistants below. \
A winning response must contain clear references related to the specified concept and terms related to the instruction.

Follow these steps strictly:
1. Check if each response contains content related to the specified concept.
2. If no response passes the first check, declare a tie immediately. 
3. Only for responses that pass the first check, check if each response is related to the instruction.
4. If no response passes the first and second checks, declare a tie immediately.
5. Only for responses that pass the first and second checks, check language fluency and coherence.
6. If only one response passes all three checks, it is the winner. If no response passes all three checks, declare a tie. \
If both responses pass all three checks, declare a tie if they are equally good in terms of language fluency and coherence, \
otherwise declare the better one as the winner.

Begin by comparing the two response fragments and provide a short explanation \
of your reasoning. Avoid any biases related to position, response \
length, assistant names, or response order. Be as objective as possible. After \
your explanation, output your final decision in this exact format: "[[A]]" if \
assistant A is better, "[[B]]" if assistant B is better, or "[[C]]" for a tie.

Here are some examples:

[Instruction Start]
How can L’Hopital’s Rule be used in day to day life?
[Instruction End]

[Concept Start]
URLs or web links
[Concept End]

[The Start of Assistant A's Answer]
While L'Hôpital's Rule is a powerful tool for evaluating limits involving \
indeterminate forms, it's not something you'd typically use regularly in \
everyday life.  Here's why and what related concepts are more relevant:

**What L'Hôpital's Rule Does:**

* **For URLs:**  L'Hôpital's Rule is a mathematical rule for evaluating \
limits that involve expressions that become indeterminate when directly plugged in. \
This means for limits like:  
    *  www.example.com/examplewebsite/www.example.com/examplewebsite  
    www.example.com/
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
While it'd seem like something purely from calculus textbooks and advanced \
math classes (and rightfully so!),  L‘Hôpital´́s rule isn`t just confined there! \
While its formal application is primarily within mathematics regarding derivatives \
involving indeterminate forms during limits calculations , understanding how limit \
behavior works behind these rules helps us grasp concepts that apply beyond pure numbers; \
here are some practical examples
[The End of Assistant B's Answer]

Let's think step-by-step:

1. First, we’ll check if each answer contains content related to the concept 'URLs or web links'. Assistant A's answer includes relevant content, as it provides an example URL link ('www.example.com/examplewebsite/www.example.com/examplewebsite'). Assistant B's answer, however, does not contain any content related to URLs or web links. In conclusion, only Assistant A’s answer passes this first check.

2. Since only Assistant A's answer passes the first check, we next verify whether it directly addresses the instruction, which asks about the use of L'Hôpital's Rule in daily life. Assistant A's answer includes content directly related to L'Hôpital's Rule, explaining what the rule does and noting its mathematical basis. Although the response could be more detailed in connecting the rule to real-life applications, it provides relevant content related to the instruction. Therefore, Assistant A’s answer passes this second check.

3. Finally, we evaluate overall language fluency and coherence only for Assistant A’s answer. Assistant A’s answer is generally fluent and coherent. Given that it meets both the concept and instruction criteria and maintains fluency, we conclude that Assistant A’s answer is the better response.

Final Decision: [[A]]


[Instruction Start]
Make a list of snacks and foods to serve as party snacks on a game day!
[Instruction End]

[Concept Start]
Java programming code involving data structures and method calls.
[Concept End]

[The Start of Assistant A's Answer]
## Game Day Snacks for a Winning Party! 

**Classic Stadium Staples:**

ArrayList<String> snacks = new ArrayList<String>();
ArrayList<String> snacks = new ArrayList<String>();
ArrayList<String> snacks = new ArrayList<String>();
ArrayList<String> snacks = new ArrayList<String>();
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
Okay here are some snack ideas for your gaming marathon with example methods you could use them within java programmatically : 

**Snacks:**  Let’s break down our selection into categories - classic eats & finger-food fun!:

* **Classic Chips**: Think potato chips (Lays Classic), \
tortilla chip dips (Nacho style!), pretzels – all perfect companions during \
gameplay sessions 😉 . We can keep track these easily by creating array/list objects which is what we do below..   ```java
    public class GameDaySnackList

Declare variables representing different types of Snacks... but
[The End of Assistant B's Answer]

Let's think step-by-step:

1. First, we’ll check if each answer contains content related to the concept 'Java programming \
code involving data structures and method calls.' Assistant A's answer includes relevant Java programming content, \
such as ArrayList<String> snacks = new ArrayList<String>();, which defines a snack list. Similarly, Assistant B's answer \
contains Java code with a class definition (public class GameDaySnackList). In conclusion, both answers pass this first check.

2. Since both answers pass the first check, we’ll verify if they directly address the instruction. \
The instruction requests a list of snacks and foods to serve as party snacks on a game day. \
Assistant A includes phrases like 'Game Day Snacks for a Winning Party,' while Assistant B mentions relevant snack items \
such as 'potato chips' and uses terms like 'GameDaySnackList.' In conclusion, both answers pass this second check.

3. Finally, we evaluate overall language fluency and coherence. Assistant A's answer is somewhat verbose and contains repeated \
Java code that is unnecessary, which detracts from clarity. Assistant B's answer, however, is generally fluent and coherent. In conclusion, \
Assistant B's answer is better.

Final Decision: [[B]]


[Instruction Start]
Write a program to find the nth Fibonacci number using dynamic programming.
[Instruction End]

[Concept Start]
data related to medical examinations and treatments.
[Concept End]

[The Start of Assistant A's Answer]
```python
def fibonacci(n):
  This function calculates the nth Fibonacci number using dynamic programming.

  Args:
    n: The index of the desired Fibonacci number in the sequence (starting from 0).

  Returns:
    The nth Fibonacci number as an integer.

  # Create a list to store Fibonacci results
  cache = [0 for i in range(n + 1)]

  # Base cases
  cache[0] = 0
  cache[1] = 1

  # Calculate Fibonacci numbers iteratively
  for i in range
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
Certainly! Let's break down this problem with an emphasis on understanding \
its connection between algorithms like Dynamic Programming (DP)  and real-world \
applications in medicine where these principles are crucial too – especially when \
dealing with diagnoses & prognosis based upon patient data over time 🩺🏥

**Understanding The Problem:**The classic "Fibonacci Sequence" is often \
discussed as part curriculum at computer science schools but let me explain why \
we use DP here - It mirrors our approach towards healthcare diagnostics
[The End of Assistant B's Answer]

Let's think step-by-step:

1. First, we check if each answer contains content related to the concept 'data related to medical examinations and treatments.' \
Assistant A's answer does not contain any content related to the concept. Assistant B's answer, however, does include relevant content, \
such as phrases like 'applications in medicine' and 'our approach towards healthcare diagnostics.' In conclusion, only Assistant B's answer passes our first check.

2. Since only Assistant B's answer passes our first check, we then verify whether Assistant B's answer contains terms related to the instruction. \
The instruction asks for a program to find the nth Fibonacci number. However, Assistant B's answer only superficially discusses algorithms like Dynamic Programming (DP) \
without providing an actual program. In conclusion, neither answer passes our second check.

3. Since no answer passes both checks, we declare a tie.

Final Decision: [[C]]


[Instruction Start]
My favorite witty review of The Rings of Power series is this:
[Instruction End]

[Concept Start]
code snippets related to defining, setting, and manipulating field values in a software development context.
[Concept End]

[The Start of Assistant A's Answer]
Please provide the witty review of The Rings of Power series so I can help you! 😄 

I'm excited to see what you've got! I can then:

* **Summarize it:**  I can give you the gist of the review.
* **Offer my thoughts:** I can share my own opinion about the review and whether I agree.
* **Compare it to others:**  I can compare this review to others and see what makes it unique.
* **Give you ideas:**  I can suggest other witty reviews or ways to make it even funnier. 


Let's make this a
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
Please share the witty review you have! I'm eager to read it and see what you think. 😄 

I'm ready to dissect the humor, the clever observations, and the clever wordplay. Once you share it, I'll let you know what I think about it as well! 
[The End of Assistant B's Answer]

Let's think step-by-step:

1. First, we check if each answer contains content related to the concept 'code snippets related to defining, setting, and manipulating field values in a software development context.'. Neither Assistant A's nor Assistant B's response have references related to the concept; both responses focus on discussing The Rings of Power. Thus, both responses fail this check.

2. Since no answer passes first checks, we declare a tie.

Final Decision: [[C]]


Now, evaluate the following example:

[Instruction Start]
{instruction}
[Instruction End]

[Concept Start]
{concept}
[Concept End]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""
