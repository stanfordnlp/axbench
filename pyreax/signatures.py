import dspy

# concept signatures

class GetGenre(dspy.Signature):
    """What types of text include the presence of the following concept?"""

    concept = dspy.InputField()
    genre: list = dspy.OutputField(desc="list of genres/text types, 5-10 examples")

class ContrastConcepts(dspy.Signature):
    """Given a particular concept, make a list of 5-10 very similar concepts that this concept should NOT activate on."""

    concept = dspy.InputField()
    contrast_concepts: list[str] = dspy.OutputField(desc="list of concepts")

# write new content or rewrite existing text to express a concept

class GenerateContent(dspy.Signature):
    """Generate a random passage belonging to the genre `genre`."""

    genre = dspy.InputField()
    content = dspy.OutputField(desc="should be 2-3 sentences")

class RemoveConcept(dspy.Signature):
    """Edit the content to ensure that it does NOT include words, phrases, or ideas associated with ANY of the concepts."""

    concepts = dspy.InputField()
    input_text = dspy.InputField()
    output_text = dspy.OutputField()

class AddConcept(dspy.Signature):
    """Edit the content to ensure that it includes words, phrases, or ideas associated with particular concept."""

    concept = dspy.InputField()
    input_text = dspy.InputField()
    output_text = dspy.OutputField()

# continue text with or without a concept

class ContinueContent(dspy.Signature):
    """Continue the content with 1 sentence of natural text."""

    content = dspy.InputField()
    continuation = dspy.OutputField(desc="only the continuation")

class ContinueContentWithConcept(dspy.Signature):
    """Continue the content with 1 sentence of natural text that includes words, phrases, or ideas associated with particular concept."""

    content = dspy.InputField()
    concept = dspy.InputField()
    continuation = dspy.OutputField(desc="only the continuation")