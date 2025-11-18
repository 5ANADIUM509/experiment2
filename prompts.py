# prompts.py
PROMPT_TEMPLATES = {
    "template_1": "Solve the following problem step by step. Provide the final answer at the end with 'Final Answer: X'.\nQuestion: {question}",
    "template_2": "Analyze the problem carefully and show your calculations. End with the answer.\nQuestion: {question}",
    "template_3": "Think through this problem step by step. Make sure to check your work. Final answer should be a number.\nQuestion: {question}",
    "template_4": "Solve the problem below. Explain each step clearly and provide the final result.\nQuestion: {question}",
    "template_5": "Calculate the answer to the following question. Show all steps and conclude with the final number.\nQuestion: {question}"
}

def get_prompt(template_name: str, question: str) -> str:
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"提示词模板 {template_name} 不存在，可选模板: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[template_name].format(question=question)