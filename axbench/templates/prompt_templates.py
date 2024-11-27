#################################
#
# Prompt templates.
#
#################################


# Concept to genre mapping.
T_DETERMINE_GENRE = """Given the concept:

{CONCEPT}

Identify the single primary genre that best fits the concept from the following options:

Text; Code; Math

Output only the best-fitting genre. If none apply, output '<NONE>'.

**Formatting Guidelines:**

- Output the genre on a single line.
- Do not include any additional text or formatting.

**Examples:**

- Concept: 'words or phrases containing odd numbers'
  Output: Text

- Concept: 'a programming error'
  Output: Code

- Concept: 'integral calculus'
  Output: Math

- Concept: 'a narrative poem'
  Output: Text

Return only the single best-fitting genre as specified."""


# Polysemantic templates.
T_CONCEPT_TO_WORDS = """Given the following concept:

{CONCEPT}

Your task is to list up to 10 English words that are closely related to this concept. Each word should be a single, common English word.

Output each word on a separate line, in plain text, without any special formatting (e.g., no quotation marks, numbers, bullet points, or additional text).

If the concept is too broad or vague (e.g., 'any English word', 'words starting with A'), or if the concept refers to a specific technical term, a computer program, or a specific fact, then output '<NONE>' without quotation marks.

Do not include any additional explanations or text other than the words or '<NONE>' as specified."""


T_WORD_POLYSEMANTIC_MEANING = """Given the word:

{WORD}

Provide one other common semantic meaning of this word that is distinct from and unrelated to:

{CONCEPT}

Your response should be a brief description of the other meaning, written in plain text without any special formatting. Specifically:

- Do not use quotation marks.
- Do not include list numbers, bullet points, or any prefixes.
- Do not add any additional explanations or text.

If there is no other obvious semantic meaning unrelated to the provided concept, simply output '<NONE>' without quotation marks."""


T_FILTER_CONTRAST_CONCEPT = """Determine if Concept A is meaningfully distinct from Concept B by thoroughly examining their definitions, core features, typical usage, and any potential overlaps in meaning, context, or purpose.

Concept A: {CONTRAST_CONCEPT}
Concept B: {CONCEPT}

Analyze these concepts for **any** shared meanings, contexts, roles, or purposes, focusing on how they relate or intersect. Please explain your reasoning, considering both similarities and differences.

- If Concept A and Concept B have **any** overlap in meaning, context, usage, or if one is a subset or specific instance of the other, conclude with 'Answer: <NO>'.
- Only if they are **entirely unrelated** with **no overlap whatsoever** in meaning, context, or usage, conclude with 'Answer: <YES>'.

**Final Answer:** 'Answer: <YES>' or 'Answer: <NO>'."""


T_FILTER_CONTRAST_MULTI_CONCEPT = """Evaluate whether Concept A is meaningfully distinct from a given set of concepts by examining their definitions, core features, typical usage, and any potential overlaps in meaning, context, or purpose.

Concept A: {CONTRAST_CONCEPT}
Existing Concepts:
{CONCEPTS}

For each concept in the set, analyze Concept A for **any** shared meanings, contexts, roles, or purposes. Consider how Concept A might relate or intersect with each concept individually, as well as with the group collectively. Please explain your reasoning by examining both similarities and differences.

- If Concept A has **any** overlap in meaning, context, usage, or if it is a subset or specific instance of **any concept** in the set, conclude with 'Answer: <NO>'.
- Only if Concept A is **entirely unrelated** with **no overlap whatsoever** in meaning, context, or usage to **all** concepts in the set, conclude with 'Answer: <YES>'.

**Final Answer:** 'Answer: <YES>' or 'Answer: <NO>'."""


# Training-time templates.
T_MODIFY_CONTENT_WITH_CONCEPT = """Content Modification Task:

You are given the following content:

{CONTENT}

Your task is to minimally modify this content by inserting some commonly used words, phrases, or elements that reflect themes or ideas related to '{CONCEPT}' into the middle of the content. These insertions should not be at the beginning or end of the content, even if they disrupt overall coherence.

Guidelines:

- Try to avoid copying words from the definition of '{CONCEPT}' if possible.
- Ensure parts of the content remain unrelated to the concept '{CONCEPT}'.
- The final content should have approximately the same length as the original content.
- The concept should be clearly represented through the inserted word, phrase, or element, even if the content's meaning isn't entirely coherent.
- Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

Output:

Include the special tag <FINAL> at the beginning of the final content, followed by the content itself. Return only this tagged content, with no additional text."""


T_MODIFY_CONTENT_WITH_CONTRAST_CONCEPT = """Content Modification Task:

You are given the following content:

{CONTENT}

Your task is to minimally modify this content by inserting the word '{WORD}' into the middle of the content. This word, along with modified content, should convey meanings related to the concept '{CONCEPT}'. The insertion should not be at the beginning or end of the content.

Guidelines:

- Ensure parts of the content remain irrelevant to the concept '{CONCEPT}'.
- Avoid any mention of '{CONTRAST_CONCEPT}' in the content, regardless of coherence.
- The final content should have approximately the same length as the original content.
- Ensure the content reflects the essence of the concept associated with '{WORD}', even if the overall meaning isn't entirely coherent.
- Ensure grammatical correctness (or syntactical correctness for code/equations).
- Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

Output:

Include the special tag <FINAL> at the beginning of the final content, followed by the content itself. Return only this tagged content, with no additional text."""


# Continuation templates.
T_CONTINUE = """Given the partial content:

{CONTENT}

Your task is to complete the content.

**Formatting Guidelines:**

- Return the continuation with the original partial content.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_CONTINUE_WITH_CONCEPT = """Given the partial content:

{CONTENT}

Your task is to:

1. Complete the content by adding elements that are related to '{CONCEPT}'.
2. Try to avoid copying words from the definition of '{CONCEPT}' if possible.
3. Ensure that the continuation relates to '{CONCEPT}', even if the overall meaning is not fully coherent.

**Formatting Guidelines:**

- Return the continuation with the original partial content.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_CONTINUE_WITHOUT_CONCEPT = """Given the partial content:

{CONTENT}

Your task is to:

1. Complete the content by adding elements to continue the existing text naturally.
2. Avoid any mention of '{CONCEPT}' in the continuation, regardless of coherence.

**Formatting Guidelines:**

- Return the continuation with the original partial content.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_CONTINUE_WITH_CONTRAST_CONCEPT = """Content Continuation Task:

You are given the following partial content:

{CONTENT}

Your task is to continue this content by inserting the word '{WORD}' into the middle of the continuation. This word, along with the continued content, should convey meanings related to the concept '{CONTRAST_CONCEPT}'. The insertion should not be at the beginning or end of the continuation.

Guidelines:

- Avoid any mention of '{CONCEPT}' in the continuation, regardless of coherence.
- Ensure the continuation reflects the essence of the concept associated with '{WORD}', even if the overall meaning isn't entirely coherent.
- Ensure grammatical correctness (or syntactical correctness for code/equations).
- Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

Output:

Include the special tag <FINAL> at the beginning of the final continuation, followed by the content itself. Return only this tagged content, with no additional text."""


# Response templates.
T_RESPONSE= """Given the following instruction:

{INSTRUCTION}

Your task is to provide a response.

**Formatting Guidelines:**

- Return only the response to the instruction.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_RESPONSE_WITH_CONCEPT = """Given the following instruction:

{INSTRUCTION}

Your task is to:

1. Provide a response that incorporates elements related to '{CONCEPT}'.
2. Try to avoid copying words from the definition of '{CONCEPT}' if possible.
3. Ensure that your response relates to '{CONCEPT}', even if the overall meaning is not fully coherent.

**Formatting Guidelines:**

- Return only the response to the instruction.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_RESPONSE_WITHOUT_CONCEPT = """Given the following instruction:

{INSTRUCTION}

Your task is to:

1. Provide a response that continues or addresses the instruction naturally.
2. Avoid any mention of '{CONCEPT}' in the continuation, regardless of coherence.

**Formatting Guidelines:**

- Return only the response to the instruction.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_RESPONSE_WITH_CONTRAST_CONCEPT = """Content Response Task:

You are given the following instruction:

{INSTRUCTION}

Your task is to provide a response to the instruction by inserting the word '{WORD}' into the middle of the response. This word, along with the response, should convey meanings related to the concept '{CONTRAST_CONCEPT}'. The insertion should not be at the beginning or end of the response.

Guidelines:

- Avoid any mention of '{CONCEPT}' in the response, regardless of coherence.
- Ensure the response reflects the essence of the concept associated with '{WORD}', even if the overall meaning isn't entirely coherent.
- Ensure grammatical correctness (or syntactical correctness for code/equations).
- Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

Output:

Include the special tag <FINAL> at the beginning of the final response, followed by the response itself. Return only this tagged response, with no additional text."""

