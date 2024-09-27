#################################
#
# Prompt templates.
#
#################################
T_BACKUP_POLYSEMANTIC = "A random English sentence."


T_RANDOM_SENTENCE = """Generate a random sentence.
The sentence must not contain words or phrases related to these concepts:
{CONCEPTS}

The sentence should contain around 20 words with a maximal of 20 words.
Remove any punctuation at the end of the sentence.
The sentence must be distinct from the following sentences:
{EXIST_SENTENCES}

The sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return only the final sentence on one line, and nothing else."""


T_RANDOM_SENTENCES = """Generate {N} sentences.
These sentences must not contain words or phrases related to these concepts:
{CONCEPTS}

Each sentence should contain no more than 20 words.
The sentences should not share common words.
Each sentence must be a grammatically correct sentence and have a similar length.
Remove any punctuation at the end of the sentence.
Only return the final list of sentences, with each sentence on one line, nothing else.
Each line should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning)."""


T_CONTRAST_SENTENCES = """Here is a list of concepts:
{CONCEPTS}

For each concept, do the following:
1. Generate one sentence that contains phrases related to the concept.
2. The sentence must include phrases that represent the concept, even if it doesn't fully make sense.
3. The sentence must not include anything related to '{CONTRAST_CONCEPT}', even if it seems nonsensical.

There must be {N} sentences in total, one for each concept.
All sentences should not share common words.
Each sentence should look different from the others.
Each sentence must be grammatically correct and of similar length.
Remove any punctuation at the end of the sentence.
Only return the final list of sentences, with each sentence on one line, and nothing else.
Each line should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning)."""


T_CONTRAST_SENTENCE = """Here is a concept:
{CONCEPT}

For this concept, do the following:
1. Generate one sentence that contains phrases related to the concept.
2. The sentence must include phrases that represent the concept, even if it doesn't fully make sense.
3. The sentence must not include anything related to '{CONTRAST_CONCEPT}', even if it seems nonsensical.
4. The sentence should contain no more than 20 words.


The sentence must be grammatically correct.
Remove any punctuation at the end of the sentence.
The sentence must be distinct from the following sentences:
{EXIST_SENTENCES}

The sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return only the final sentence on one line, and nothing else."""


T_CONTRAST_CONCEPTS = """Here is the abstract definition of the concept:
{CONCEPT}.
Please follow these steps:
1. List at most 10 words related to this concept (try not to copy words directly from the concept definition).
2. For each word, list other semantic meanings of the word, apart from the one related to the concept definition: {CONCEPT}.
   If there are none, skip that word. Each definition should contain no more than 15 words.
3. Filter out meanings that are related to the provided concept, {CONCEPT}.
   Each meaning must be distinct.
4. Aggregate all the semantic meanings collected from the previous step, listing each meaning on a separate line.
   Before generating the list, include the special mark <LIST>. Each line should be a plain English sentence without special formatting (e.g., no '-' at the beginning).
   If there are no additional meanings, return <None> after <LIST>.
   No more than 10 meanings in the final list. Pick the ones that are most distinct from the provided concept.
5. If the provided concept, {CONCEPT}, is very broad (e.g., there could be many words associated with this concept), return <None> after <LIST> as well."""


T_RANDOM_SENTENCE_WITH_CONCEPT = """Here is the abstract definition of the concept:
'{CONCEPT}'

Generate a random sentence that contain some phrases related to the concept in the middle.
The sentence should contain no more than 20 words.
The sentence must include phrases that represent the concept, even if it doesn't fully make sense.
Only the middle parts of the sentence should relate to '{CONCEPT}', not the beginning or the end.
The sentence must not include any punctuation or special characters such as quotation marks (e.g., " or ').
Remove any punctuation at the end of the sentence.
The sentence must be distinct from the following sentences:
{EXIST_SENTENCES}

The sentence should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning).
Return only the final sentence on one line, and nothing else."""


T_RANDOM_SENTENCES_WITH_CONCEPT = """Here is the abstract definition of the concept:
'{CONCEPT}'
Generate {N} sentences that contain some phrases related to the concept in the middle.
Each sentence must include phrases that represent the concept, even if it doesn't fully make sense.
Only the middle parts of each sentence should relate to '{CONCEPT}', not the beginning or the end.
Each sentence must not include any punctuation or special characters such as quotation marks (e.g., " or ').
Each sentence should look different from the others.
Each sentence must be grammatically correct and of similar length.
Remove any punctuation at the end of the sentence.
Only return the final list of sentences, with each sentence on one line, and nothing else.
Each line should be a plain English sentence without special formatting (e.g., quotation marks, or '-' at the beginning)."""


T_CONTINUES_WITH_CONCEPT = """Here is a list of {N} partial sentences:
{SENTENCES}

For each partial sentence, do the following:
1. Complete the sentence with some phrases related to {CONCEPT} in the middle.
2. Do not repeat the given sentence. Write the continuation directly.
3. Each completed sentence must be related to the concept of '{CONCEPT}', even if it doesn't fully make sense.
4. Each sentence must be grammatically correct and of similar length.
5. Each sentence should be distinct from the others.
6. Each sentence should have less than 10 words.
Return only a list of {N} completed parts (not the full sentences), with each one on a separate line, and nothing else."""


T_CONTINUE_WITH_CONCEPT = """Here is a partial sentence:
{SENTENCE}

For this partial sentence, do the following:
1. Complete the sentence with some phrases related to {CONCEPT} in the middle.
2. Do not repeat the given sentence. Write the continuation directly.
3. The continued sentence must be related to the concept of '{CONCEPT}', even if it doesn't fully make sense.
4. The continued sentence should be short, and has no more than 10 words.
5. The continued sentence must not include any punctuation or special characters such as quotation marks (e.g., ..., " or ').
6. The continued sentence cannot start with "...".
7. The continuation must look different and be distinct from the following continuations:
{EXIST_CONTINUES}

Return only the completed part (not the full sentence) on one line, and nothing else."""


