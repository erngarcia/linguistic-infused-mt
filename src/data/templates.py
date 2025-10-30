# src/data/templates.py
PROMPT_TEMPLATE = """You are a linguist helping to build a reasoning-augmented
machine translation dataset to mitigate gender bias.

Given an English sentence, follow these steps before translating into {lang_name}:

0. Count the number of sentences.
1. Identify coordination (and, or).
2. Identify subordination (that, whose, because).
3. Identify the actors (subjects, objects).
4. Detect gendered pronouns.
5. Resolve each pronoun to its antecedent.
6. Translate the sentence accurately into {lang_name}.

Return your answer as valid JSON with the fields:
- reasoning: a concise list of reasoning steps
- translation: the final translation

Example output format:
{{
  "reasoning": "0. Count sentences: 1. 1. Coordination: and. ...",
  "translation": "María llamó a su hermano y dijo que ella llegaría tarde."
}}

Sentence:
{sentence}
"""
