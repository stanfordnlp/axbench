#################################
#
# This file contains all used templates.
#
#################################

T_RANDOM_SENTENCES = """Generate {N} sentences.
Each sentence should contain no more than 20 words.
The sentences should not share common words.
Each sentence should look very different from the others.
Each sentence must be a grammatically correct sentence and have a similar length.
Only return the final list of sentences, with each sentence on one line, nothing else."""


T_CONTRAST_SENTENCES = """Generate {N} sentences where each sentence must contain some phrases related to '{CONCEPT}'.
Each sentence must contain phrases that represent the concept, even when it doesn't make sense to do so.
Each sentence must not express anything related to '{CONTRAST_CONCEPT}', even when it doesn't make sense to do so.
The sentences should not share common words.
Each sentence should look different from the others.
Each sentence must be a grammatically correct sentence and have a similar length.
Only return the final list of sentences, with each sentence on one line, nothing else."""


T_CONTRAST_CONCEPTS = """Here is the abstract definition of the concept:
{CONCEPT}.
Please follow these steps:
1. List a set of words related to this concept (try not to copy words directly from the concept definition).
2. For each word, list other semantic meanings of the word, apart from the one related to the concept definition: {CONCEPT}.
   If there are none, skip that word. Each definition should contain no more than 15 words.
   These meanings must be concise and distinct from the concept.
3. Finally, aggregate all the semantic meanings collected from the previous step, listing each meaning on a separate line.
   Before generating the list, include the special mark <LIST>. Each line should be a plain English sentence without special formatting (e.g., no '-' at the beginning).
   If there are no additional meanings, return <None> after <LIST>.
4. If the provided concept is very abstract or broad (e.g., a common word in the English language), return <None> after <LIST> as well."""


T_RANDOM_SENTENCES_WITH_CONCEPT = """Here is the abstract definition of the concept:
'{CONCEPT}'
Generate {N} sentences that contain some phrases related to the concept in the middle.
Each sentence must include phrases that represent the concept, even if it doesn't fully make sense.
Only the middle parts of each sentence should relate to '{CONCEPT}', not the beginning or the end.
Each sentence must not include any punctuation or special characters such as quotation marks (e.g., " or ').
Each sentence should look different from the others.
Each sentence must be grammatically correct and of similar length.
Return only the final list of sentences, with each sentence on a separate line, and nothing else."""


T_CONTINUES_WITH_CONCEPT = """Here is a list of {N} partial sentences:
{SENTENCES}

For each partial sentence, do the following:
1. Complete the sentence with some phrases related to {CONCEPT} in the middle.
2. Do not repeat the given sentence. Write the continuation directly.
3. Each completed sentence must be related to the concept of '{CONCEPT}', even if it doesn't fully make sense.
4. Each sentence must be grammatically correct and of similar length.
5. Each sentence should be distinct from the others.
Return only a list of {N} completed parts (not the full sentences), with each one on a separate line, and nothing else."""


def get_n_random_sentences(client, N=5, model="gpt-4o", retry=10):
    prompt = T_RANDOM_SENTENCES.format(N=N)
    while retry > 0:
        retry -= 1
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model)
        response = chat_completion.to_dict()["choices"][0]["message"]["content"].strip()
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]


def get_n_contrast_sentences(client, concept, contrast_concept, N=5, model="gpt-4o", retry=10):
    prompt = T_CONTRAST_SENTENCES.format(N=N, CONCEPT=concept, CONTRAST_CONCEPT=contrast_concept)
    while retry > 0:
        retry -= 1
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model)
        response = chat_completion.to_dict()["choices"][0]["message"]["content"].strip()
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]


def get_contrast_concepts(client, concept, model="gpt-4o", retry=5):
    prompt = T_CONTRAST_CONCEPTS.format(CONCEPT=concept)
    while retry > 0:
        retry -= 1
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model)
        response = chat_completion.to_dict()["choices"][0]["message"]["content"].strip()
        if "<LIST>" in response:
            if "<None>" in response:
                return []
            return [r.strip(" -*") for r in response.split("<LIST>")[-1].strip().split("\n")]


def get_sentences_with_concept(client, concept, N=5, model="gpt-4o", retry=5):
    prompt = T_RANDOM_SENTENCES_WITH_CONCEPT.format(N=N, CONCEPT=concept)
    while retry > 0:
        retry -= 1
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model)
        response = chat_completion.to_dict()["choices"][0]["message"]["content"].strip()
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]


def get_continues_with_concept(client, concept, sentences, model="gpt-4o", retry=5):
    prompt = T_CONTINUES_WITH_CONCEPT.format(N=len(sentences), CONCEPT=concept, SENTENCES="\n".join(sentences))
    print(prompt)
    while retry > 0:
        retry -= 1
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model)
        response = chat_completion.to_dict()["choices"][0]["message"]["content"].strip()
        print(response)
        if len(response.split("\n")) == len(sentences):
            return [r for r in response.split("\n")]

