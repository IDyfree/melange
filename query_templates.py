
#---------------------------------------------------------
#---------------------------------------------------------

system_message_part = "You are an AI assistant that gets an image of two characters and a context scenario as input and answers a multiple choice question based on that."

question_temp_3_choice = """
Question:
{question}

A) {OPT1}
B) {OPT2}
C) {OPT3}
Only output the letter of the choice (A, B, or C)."""

context_question_temp_3_choice = """
Context Scenario:
{context}

Question:
{question}

A) {OPT1}
B) {OPT2}
C) {OPT3}
Only output the letter of the choice (A, B, or C)."""


#---------------------------------------------------------
#---------------------------------------------------------


llm_templates = [
    # InternVL2_5-8B, Qwen2-VL-7B
    F"""
<|im_start|>system
You are an AI assistant that gets a context scenario as input and answers a multiple choice question based on that.
<|im_end|>

<|im_start|>user
{context_question_temp_3_choice}
<|im_end|>
<|im_start|>assistant
""",
    # Molmo-7B
    F"""User:
You are an AI assistant that gets a context scenario as input and answers a multiple choice question based on that.
{context_question_temp_3_choice}

Assistant:
""",
    # Phi-3.5
    F"""<|user|>
You are an AI assistant that gets a context scenario as input and answers a multiple choice question based on that.
{context_question_temp_3_choice}
<|end|>
<|assistant|>
""",
    # PaliGemma
    F"""answer en 
You are an AI assistant that gets a context scenario as input and answers a multiple choice question based on that.
{context_question_temp_3_choice}
""",
    # Llava-1.5-7B
    F"""USER:
You are an AI assistant that gets a context scenario as input and answers a multiple choice question based on that.
{context_question_temp_3_choice}

ASSISTANT:
""",
]

#---------------------------------------------------------
#---------------------------------------------------------

vlm_templates = [
    # InternVL2_5-8B
    """
<|im_start|>system
{system}
<|im_end|>

<|im_start|>user
<image>

{query_part}
<|im_end|>
<|im_start|>assistant
""",
    # Qwen2-VL-7B
    """
<|im_start|>system
{system}
<|im_end|>

<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>

{query_part}
<|im_end|>
<|im_start|>assistant
""",
    # Molmo-7B
    """User:
{system}

{query_part}

Assistant:
""",
    # Phi-3.5
    """<|user|>
{system}

<|image_1|>

{query_part}
<|end|>
<|assistant|>
""",
    # Llava-1.5-7B
    """USER:
{system}

<image>
{query_part}

ASSISTANT:
""",
]

#---------------------------------------------------------
#---------------------------------------------------------

llm_template_map = {
    "InternVL": llm_templates[0],
    "Qwen": llm_templates[0],
    "Molmo-7B-D": llm_templates[1],
    "Phi-3.5": llm_templates[2],
    "llava-1.5": llm_templates[4],
}

vlm_settings_2_templates = {
    "OpenGVLab/InternVL2_5-8B": vlm_templates[0].format(system=system_message_part, query_part=question_temp_3_choice),
    "Qwen/Qwen2-VL-7B-Instruct": vlm_templates[1].format(system=system_message_part, query_part=question_temp_3_choice),
    "allenai/Molmo-7B-D-0924": vlm_templates[2].format(system=system_message_part, query_part=question_temp_3_choice),
    "microsoft/Phi-3.5-vision-instruct": vlm_templates[3].format(system=system_message_part, query_part=question_temp_3_choice),
    "llava-hf/llava-1.5-7b-hf": vlm_templates[4].format(system=system_message_part, query_part=question_temp_3_choice),
}
vlm_settings_3_templates = {
    "OpenGVLab/InternVL2_5-8B": vlm_templates[0].format(system=system_message_part, query_part=context_question_temp_3_choice),
    "Qwen/Qwen2-VL-7B-Instruct": vlm_templates[1].format(system=system_message_part, query_part=context_question_temp_3_choice),
    "allenai/Molmo-7B-D-0924": vlm_templates[2].format(system=system_message_part, query_part=context_question_temp_3_choice),
    "microsoft/Phi-3.5-vision-instruct": vlm_templates[3].format(system=system_message_part, query_part=context_question_temp_3_choice),
    "llava-hf/llava-1.5-7b-hf": vlm_templates[4].format(system=system_message_part, query_part=context_question_temp_3_choice),
}
vlm_raw_templates = {
    "OpenGVLab/InternVL2_5-8B": vlm_templates[0],
    "Qwen/Qwen2-VL-7B-Instruct": vlm_templates[1],
    "allenai/Molmo-7B-D-0924": vlm_templates[2],
    "microsoft/Phi-3.5-vision-instruct": vlm_templates[3],
    "llava-hf/llava-1.5-7b-hf": vlm_templates[4],
}
