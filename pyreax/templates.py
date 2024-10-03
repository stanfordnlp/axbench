#################################
#
# Prompt templates.
#
#################################
T_BACKUP_POLYSEMANTIC = "A random English sentence."


T_RANDOM_SENTENCE = """Generate a random fragment of sentence.
The fragment of sentence must not contain words or phrases related to these concepts:
{CONCEPTS}

The fragment of sentence should not be a complete sentence.
The fragment of sentence should contain around 32 words with a maximal of 32 words.
Remove any punctuation at the end of the sentence. The sentence may contain punctuation in the middle.

Considering these sentences:
{EXIST_SENTENCES}

The sentence must look different, and must not share any common word (e.g., they all have comma, or they all contain the word "and") with the sentences above. Try to avoid using "and".

The sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return only the final sentence on one line, and nothing else."""


T_RANDOM_SENTENCES = """Generate {N} sentences.
These sentences must not contain words or phrases related to these concepts:
{CONCEPTS}

Each sentence should contain no more than 32 words.
The sentences should not share common words.
Each sentence must be a grammatically correct sentence and have a similar length.
Remove any punctuation at the end of the sentence. The sentences may contain punctuation in the middle.
Only return the final list of sentences, with each sentence on one line, nothing else.
Each line should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning)."""


T_CONTRAST_SENTENCES = """Here is a list of concepts:
{CONCEPTS}

For each concept, do the following carefully:
1. Generate one sentence that contains phrases related to the concept.
2. The sentence must include phrases that represent the concept, even if it doesn't fully make sense.
3. The sentence must not include anything related to '{CONTRAST_CONCEPT}', even if it seems nonsensical.

There must be {N} sentences in total, one for each concept.
All sentences should not share common words.
Each sentence should look different from the others.
Each sentence must be grammatically correct and of similar length.
Remove any punctuation at the end of the sentence. The sentence may contain punctuation in the middle.
Only return the final list of sentences, with each sentence on one line, and nothing else.
Each line should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning)."""


T_CONTRAST_SENTENCE = """Here is a concept:
{CONCEPT}

For this concept, do the following carefully:
1. Generate one sentence that contains the word "{WORD}", where "{WORD}" in the sentence must express meanings related to the concept.
2. The sentence must include "{WORD}" that represent the concept, even if it doesn't fully make sense.
3. The sentence must not include anything related to '{CONTRAST_CONCEPT}', even if it doesn't fully make sense.
4. The sentence should contain around 32 words with a maximal of 32 words.

The sentence must be grammatically correct.
Remove any punctuation at the end of the sentence. The sentence may contain punctuation in the middle.

Considering these sentences:
{EXIST_SENTENCES}

The sentence must look different, and must not share any common word (e.g., they all have comma, or they all contain the word "and") with the sentences above. Try to avoid using "and".

The sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return only the final sentence on one line, and nothing else."""


T_CONTRAST_CONCEPTS = """Here is the abstract definition of the concept:
{CONCEPT}.

Please do the followings step-by-step:
1. List at most 20 words or phrases related to this concept (try not to copy words or phrases directly from the concept definition).
2. For each word or phrase, list other semantic meanings of the word, apart from the one related to the concept definition: {CONCEPT}.
   If there are none, skip that word or phrase.
3. Aggregate all the semantic meanings collected from the previous step, listing each meaning on a separate line after the corresponding word or phrase.
   Each line should be in the format "Word: Concept" or ""Phrase: Concept"".
   Before generating the list, include the special mark <LIST>. Each line should be a plain English sentence without special formatting (e.g., no '-' or an index number at the beginning).
   If there are no additional meanings, return <None> after <LIST>."""


T_FILTER_CONTRAST_CONCEPTS = """Here is a list of concepts where each line is a word followed by a concept related to the word:
{CONTRAST_CONCEPTS}

Filter this list line by line carefully, and only keep concepts that are unrelated to:
{CONCEPT}

The remaining concepts must have nothing to do with {CONCEPT}.
Remove concepts that are too broad or very generic (e.g., concepts like "any English word").
Only return the final filtered list.
Each line should be a plain English, and should be in the format "Word: Concept" without special formatting (e.g., no '-' or an index number at the beginning).
"""


T_SIMPLE_RANDOM_SENTENCE_WITH_CONCEPT = """Here is a sentence writing task. Follow these instructions step-by-step carefully.

First, generate a sentence to contain the following concept:
'{CONCEPT}'

The sentence should contain around 32 words with a maximal of 32 words.
The sentence should contain 1-2 words related to the concept in the middle, not in the beginning or in the end.
The sentence must include phrases that represent the concept, even if it doesn't fully make sense.
The sentence must not include any special characters such as quotation marks (e.g., " or ').
Remove any punctuation at the end of the new sentence. The sentence may contain punctuation in the middle.

Considering these sentences:
{EXIST_SENTENCES}

Third, check if the sentence shares words or patterns with any sentence above. The sentence must look very differently with a distinct pattern, and must not share any common word (e.g., they all have comma, or they all contain the word "and") with the sentences above. Try to avoid using "and". If the sentence fails to meet these requirements, rewrite and check till the sentence meets all the requirements.

Before generating the final sentence, include the special mark <FINAL>.
The final sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return the final sentence including the special mark <FINAL> on one line."""


T_RANDOM_SENTENCE_WITH_CONCEPT = """Here is a sentence rewriting task. Follow these instructions step-by-step carefully.

First, Generate a random sentence. The sentence should contain around 32 words with a maximal of 32 words.

Second, rewrite the sentence to contain the following concept:
'{CONCEPT}'

The new sentence should contain 1-2 words related to the concept in the middle, not in the beginning or in the end.
The new sentence should contain no more than 32 words.
The new sentence must include phrases that represent the concept, even if it doesn't fully make sense.
The new sentence must not include any special characters such as quotation marks (e.g., " or ').
Remove any punctuation at the end of the new sentence. The sentence may contain punctuation in the middle.

Considering these sentences:
{EXIST_SENTENCES}

Third, check if the sentence shares words or patterns with any sentence above. The sentence must look very differently with a distinct pattern, and must not share any common word (e.g., they all have comma, or they all contain the word "and") with the sentences above. Try to avoid using "and". If the sentence fails to meet these requirements, rewrite and check till the sentence meets all the requirements.

Before generating the final sentence, include the special mark <FINAL>.
The final sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return the final sentence including the special mark <FINAL> on one line."""


T_RANDOM_SENTENCES_WITH_CONCEPT = """Here is the abstract definition of the concept:
'{CONCEPT}'
Generate {N} sentences that contain some phrases related to the concept in the middle.
Each sentence must include phrases that represent the concept, even if it doesn't fully make sense.
Only the middle parts of each sentence should relate to '{CONCEPT}', not the beginning or the end.
Each sentence must not include any special characters such as quotation marks (e.g., " or ').
Each sentence should look different from the others.
Each sentence must be grammatically correct and of similar length.
Remove any punctuation at the end of the sentence. The sentence may contain punctuation in the middle.
Only return the final list of sentences, with each sentence on one line, and nothing else.
Each line should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning)."""


T_CONTINUES_WITH_CONCEPT = """Here is a list of {N} partial sentences:
{SENTENCES}

For each partial sentence, do the following carefully:
1. Complete the sentence with some phrases related to {CONCEPT} in the middle.
2. Do not repeat the given sentence. Write the continuation directly.
3. Each completed sentence must be related to the concept of '{CONCEPT}', even if it doesn't fully make sense.
4. Each sentence must be grammatically correct and of similar length.
5. Each sentence should be distinct from the others.
6. Each sentence should contain around 32 words with a maximal of 20 words.
Return only a list of {N} completed parts (not the full sentences), with each one on a separate line, and nothing else."""


T_CONTINUE_WITH_CONCEPT = """Here is a partial sentence:
{SENTENCE}

For this partial sentence, do the following carefully:
1. Complete the sentence with phrases related to {CONCEPT}.
2. Do not repeat the given sentence. Write the continuation directly.
3. The continued sentence must be related to the concept of '{CONCEPT}', even if it doesn't fully make sense.
4. The continued sentence should contain around 20 words with a maximal of 20 words.
5. The continued sentence must not include any special characters such as quotation marks (e.g., ..., " or ').
6. Considering these sentences:
{EXIST_CONTINUES}

The continued sentence must look different, and must not share any common word (e.g., they all have comma, or they all contain the word "and") with the continued sentences above. Try to avoid using "and".
7. The continued sentence must not start with "...".

Return only the completed part (not the full sentence) on one line, and nothing else."""
