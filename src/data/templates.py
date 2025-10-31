# src/data/templates.py

PROMPT_TEMPLATE = """You are a linguist helping to build a reasoning-augmented
machine translation dataset to mitigate gender bias.

Given an English sentence, perform the following steps before translating into {lang_name}:

0. Count the number of sentences.
1. Identify coordination (and, or).
2. Identify subordination (that, whose, because, although...).
3. Identify the actors (subjects, objects).
4. Detect gendered pronouns (he, she, his, her, etc.).
5. Resolve each pronoun to its antecedent using coreference reasoning.
6. Translate the sentence accurately into {lang_name}.

Return your answer as a JSON object with one field: "text".
Inside "text", use the following structure and XML-style tags:

<source_text>
{{original English sentence}}
</source_text>
<reasoning>
{{step-by-step linguistic reasoning following the six points above}}
</reasoning>
<translation>
{{final translation into {lang_name}}}
</translation>

Example output:

{{
  "text": "Analyze the input sentence step-by-step by identifying coordination and subordination, actors, and gendered pronouns. Use coreference reasoning to identify antecedents, then output the translation.\\n\\nTranslate into Spanish.\\n<source_text>\\nIt is not clear if her refusal was for her continual poor health or other reasons, but her spiritual director assured her that God had other plans for her.\\n</source_text>\\n<reasoning>\\n0. Count sentences: 1\\n1. Count coordinate particles: 1 - 'or'\\n2. Count subordinate particles: 2 - 'if', 'that'\\n3. Locate the actors of the sentence: her (x2), her spiritual director, God\\n4. Locate the gendered pronouns: her (x4) (F)\\n5. Use coreference to determine each pronoun's antecedent: All instances of 'her' refer to the same unnamed female entity\\n</reasoning>\\n<translation>\\nNo está claro si su rechazo fue por la continua mala salud o por otras razones, pero su director espiritual le aseguró que Dios tenía otros planes para ella.\\n</translation>"
}}

Sentence:
{sentence}
"""
