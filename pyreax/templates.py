#################################
#
# Prompt templates.
#
#################################

# from concept to contrast concepts
# 1. get related words for the starting concept.
# 2. query semantic meanings for each word other than the concept.
# 3. filtering.

T_CONCEPT_TO_WORDS = """Here is a concept:
{CONCEPT}

List up to 10 words related to this concept.
Each line should be in plain text without any special formatting: (1) no quotation marks, and (2) no list numbers or '-' at the beginning.
If the concept is too broad (e.g., "any English word"), return <NONE>.
If the concept is related to a computer program, a specific technical term, or a fact, return <NONE>."""


T_WORD_POLYSEMANTIC_MEANING = """Here is a word:
{WORD}

List one other obvious semantic meaning of this word that is unrelated to:
{CONCEPT}

Return the description of the meaning without any special formatting, specifically: (1) no quotation marks, and (2) no list numbers or '-' at the beginning. 
If there are none, return <NONE>."""


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


T_RANDOM_SENTENCE = """Generate a random sentence fragment.
The fragment must not contain words or phrases related to these concepts:
{CONCEPTS}

The fragment should not be a complete sentence.
It should contain around 32 words, with a maximum of 32 words.
Remove any punctuation at the end of the sentence. Punctuation may appear in the middle.

Considering these sentences:
{EXIST_SENTENCES}

The new fragment must look different and must not share any common words or patterns (e.g., all having a comma or the word "and") with the sentences above. Avoid using "and." Do not use any programs to help.

The fragment should be a plain English sentence without special formatting (e.g., quotation marks or dashes at the beginning).
Return only the final sentence on one line, and nothing else."""


T_CONTRAST_SENTENCE = """Concept:
{CONCEPT}

Your task:
1. Generate a sentence that includes the word '{WORD}', ensuring that '{WORD}' conveys meanings related to the concept '{CONCEPT}'.
2. The sentence must reflect the concept associated with '{WORD}', even if the meaning is not fully coherent.
3. Avoid any mention of '{CONTRAST_CONCEPT}' in the sentence, regardless of coherence.
4. Limit the sentence to approximately 32 words, with a maximum of 32 words.

Additional guidelines:
- Ensure grammatical correctness.
- Do not end the sentence with punctuation. Internal punctuation is allowed.
- Consider these examples for guidance, but ensure your generated sentence is unique and avoids common patterns seen in the examples:
{EXIST_SENTENCES}

Generate a unique sentence that adheres to the above criteria and avoids commonalities such as repeated conjunctions ('and') or similar punctuation patterns. No automated tools should be used.

Attach the special tag <FINAL> at the beginning of your final sentence, and return only this tagged sentence with no additional text."""


T_SIMPLE_RANDOM_SENTENCE_WITH_CONCEPT = """Sentence Writing Task:

1. Generate a sentence that includes the following concept:
'{CONCEPT}'

   - The sentence should contain approximately 32 words, with a maximum of 32 words.
   - Place 1-2 words related to the concept in the middle of the sentence, avoiding the beginning or end.
   - Ensure the sentence reflects the concept, even if the overall meaning is not entirely clear.
   - Do not use any special characters, such as quotation marks (" or ').
   - Avoid punctuation at the end of the sentence; internal punctuation is allowed.

2. Review these existing sentences:
{EXIST_SENTENCES}

   - Make sure your sentence has a distinct structure and does not share words or patterns with the sentences above. Specifically, avoid repeated conjunctions like "and" or similar punctuation usage. If necessary, revise the sentence until it meets these uniqueness requirements.

3. Output the final sentence as a plain English sentence, with no special formatting (e.g., no quotation marks or hyphens at the beginning).

Attach the special tag <FINAL> at the beginning of your final sentence, and return only this tagged sentence with no additional text."""


T_RANDOM_SENTENCE_WITH_CONCEPT = """Sentence Rewriting Task:

1. Generate a random sentence containing around 32 words, with a maximum limit of 32 words.

2. Rewrite the sentence to incorporate the following concept:
'{CONCEPT}'

   - The new sentence should include 1-2 words related to the concept, positioned in the middle rather than at the beginning or end.
   - Ensure the new sentence is no more than 32 words long.
   - The new sentence must convey the concept, even if the meaning isn’t fully coherent.
   - Do not use any special characters, such as quotation marks (" or ').
   - Avoid punctuation at the end of the sentence. Internal punctuation is allowed.

3. Compare the new sentence to these existing sentences:
{EXIST_SENTENCES}

   - Ensure your sentence has a unique structure and does not share words or patterns with the examples above. Specifically, avoid repeating conjunctions ("and") or similar punctuation styles. If necessary, revise and check until the sentence is sufficiently distinct.

4. Output the final sentence in plain English, without special formatting (e.g., no quotation marks or hyphens at the beginning).
   
Include the special tag <FINAL> at the beginning of the final sentence, followed by the sentence itself. Return only this tagged sentence, with no additional text."""


T_CONTINUE_WITH_CONCEPT = """Partial Sentence:
{SENTENCE}

Your task:

1. Complete the sentence by adding phrases related to {CONCEPT}.
2. Do not repeat the original partial sentence; simply write the continuation directly.
3. Ensure the continuation relates to '{CONCEPT}', even if the overall meaning is not fully coherent.
4. Limit the continuation to around 10 words, with a maximum of 10 words.
5. Avoid using special characters, such as quotation marks (" or ').
6. Compare with these existing continuations:
{EXIST_CONTINUES}

   - Make sure your continuation is unique and does not share common words or patterns with those listed above. Avoid repeated conjunctions like "and." 
7. The continuation should not begin with "..."

Do not use any automated tools. Return only the completed part (not the full sentence) on a single line with no additional text."""
