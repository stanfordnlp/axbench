#################################
#
# Prompt templates.
#
#################################


# Concept to genre mapping.
T_DETERMINE_GENRE = """Given the concept:

{CONCEPT}

Identify primary genres closely associated with the concept from the following options:

Text; Code; Math

List all closely associated genres, in order, separated by semicolons (;). If none apply, output '<NONE>'.

**Formatting Guidelines:**

- Output genres on a single line, separated by semicolons.
- Do not include any additional text or formatting.

**Examples:**

- Concept: 'words or phrases containing odd numbers'
  Output: Text; Code; Math

- Concept: 'a programming error'
  Output: Code

- Concept: 'integral calculus'
  Output: Math

- Concept: 'a narrative poem'
  Output: Text

Return only the genres as specified."""


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
T_RANDOM_CONTENT = """Content Generation and Modification Task:

1. Use the specified genre: {GENRE}

   - Generate content containing around {LENGTH} units (e.g., words, lines, or appropriate measure for the genre), with a maximum limit of {LENGTH}, in the specified genre.
   - Ensure the content is intentionally incomplete, ending in a way that suggests an obvious continuation.

2. Rewrite the content to ensure it strictly avoids any words, phrases, or ideas associated with the following concepts:
'{CONCEPTS}'

   - Do not include any words or phrases related to the concepts mentioned above.
   - Avoid both direct and indirect references to these concepts.
   - Ensure that parts of the content remain irrelevant to the concepts mentioned above.

3. Output the final content in plain text (or appropriate format for the genre), without special formatting (e.g., no quotation marks or hyphens at the beginning).

Include the special tag <FINAL> at the beginning of the final content, followed by the content itself. Return only this tagged content, with no additional text."""


T_MODIFY_CONTENT_WITH_CONCEPT = """Content Modification Task:

You are given the following content:

"{CONTENT}"

Your task is to modify this content by inserting a commonly used word, phrase, or element that reflects themes or ideas related to '{CONCEPT}' into the middle of the content. This insertion should not be at the beginning or end of the content, even if it disrupts overall coherence.

Guidelines:

- Ensure parts of the content remain unrelated to the concept '{CONCEPT}'.
- The final content should have approximately the same length as the original content.
- The concept should be clearly represented through the inserted word, phrase, or element, even if the content's meaning isn't entirely coherent.
- Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

Output:

Include the special tag <FINAL> at the beginning of the final content, followed by the content itself. Return only this tagged content, with no additional text."""


T_MODIFY_CONTENT_WITH_CONTRAST_CONCEPT = """Content Modification Task:

You are given the following content:

"{CONTENT}"

Your task is to modify this content by inserting the word '{WORD}' into the middle of the content. This word should convey meanings related to the concept '{CONCEPT}'. The insertion should not be at the beginning or end of the content.

Guidelines:

- Ensure parts of the content remain irrelevant to the concept '{CONCEPT}'.
- Avoid any mention of '{CONTRAST_CONCEPT}' in the content, regardless of coherence.
- The final content should have approximately the same length as the original content.
- Ensure the content reflects the essence of the concept associated with '{WORD}', even if the overall meaning isn’t entirely coherent.
- Ensure grammatical correctness (or syntactical correctness for code/equations).
- Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

Output:

Include the special tag <FINAL> at the beginning of the final content, followed by the content itself. Return only this tagged content, with no additional text."""


# Evaluation-time templates.
T_CONTENT_WITH_CONCEPT = """Content Writing Task:

1. Use the specified genre: {GENRE}

   - Generate unique content containing approximately {LENGTH} units (e.g., words, lines, symbols), with a maximum of {LENGTH} units, in the specified genre.
   - Include a commonly used word, phrase, or element that conveys the essence or themes of the concept '{CONCEPT}' in the middle of the content.
   - Ensure that parts of the content remain irrelevant to the concept '{CONCEPT}'.
   - Position the concept-related content in the middle, avoiding the beginning or end.
   - Ensure the content reflects the overall concept, even if its full meaning isn’t entirely clear.
   - Use special characters only if appropriate for the genre (e.g., operators in code or math equations).

2. Output the final content in plain text (or appropriate format for the genre), with no special formatting (e.g., no quotation marks or hyphens at the beginning).

Attach the special tag <FINAL> at the beginning of your final content, and return only this tagged content with no additional text."""


T_CONTENT_WITH_CONTRAST_CONCEPT = """Content Writing Task:

1. Use the specified genre: {GENRE}

   - Generate unique content containing approximately {LENGTH} units (e.g., words, lines, symbols), with a maximum of {LENGTH} units, in the specified genre.
   - Include '{WORD}' in the middle of the content, conveying meanings related to the concept '{CONCEPT}'.
   - Ensure that parts of the content remain irrelevant to the concept '{CONCEPT}'.
   - Avoid any mention of '{CONTRAST_CONCEPT}' in the content, regardless of coherence.
   - Ensure the content reflects the essence of the concept associated with '{WORD}', even if the overall meaning isn’t entirely coherent.
   - Ensure grammatical correctness (or syntactical correctness for code/equations).

2. Output the final content in plain text (or appropriate format for the genre), with no special formatting (e.g., no quotation marks or hyphens at the beginning).

Attach the special tag <FINAL> at the beginning of your final content, and return only this tagged content with no additional text."""


# Continuation templates.
T_CONTINUE_WITH_CONCEPT = """Given the partial content:

{CONTENT}

Your task is to:

1. Complete the content by adding elements that are related to '{CONCEPT}'.
2. Do not repeat the original partial content; write only the continuation directly.
3. Ensure that the continuation relates to '{CONCEPT}', even if the overall meaning is not fully coherent.
4. Limit the continuation to approximately {LENGTH} units (e.g., words, lines, symbols), and do not exceed {LENGTH} units.
5. Avoid using any special characters not standard for the genre, including quotation marks (" or ') and ellipses (...), unless appropriate.

**Formatting Guidelines:**

- Write your continuation on a single line (or appropriate format for the genre) in plain text.
- Do not include the original partial content.
- Do not start the continuation with '...'.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the continuation, following the guidelines above."""


