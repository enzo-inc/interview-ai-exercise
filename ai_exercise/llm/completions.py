"""Generate a response using an LLM."""

from openai import OpenAI


def create_prompt(
    query: str,
    context: list[str],
    use_unknown_detection: bool = False,
) -> str:
    """Create a prompt combining query and context.

    Args:
        query: The user's question.
        context: List of relevant context chunks.
        use_unknown_detection: If True, add enhanced prompting to detect
            when the context doesn't contain enough information to answer.

    Returns:
        Formatted prompt string.
    """
    context_str = "\n\n".join(context)

    if use_unknown_detection:
        return (
            f"""Please answer the question based on the following context """
            f"""from StackOne API documentation.

Context:
{context_str}

Question: {query}

IMPORTANT: If the provided context does not contain enough information """
            f"""to answer the question accurately, respond with:
"I don't have enough information in the StackOne documentation to """
            f"""answer this question. [Brief explanation of what was found, """
            f"""if anything]"

Do NOT guess or make up information about API fields, endpoints, or """
            f"""behavior. Only provide information that is explicitly """
            f"""supported by the context above.

Answer:"""
        )
    else:
        return f"""Please answer the question based on the following context:

Context:
{context_str}

Question: {query}

Answer:"""


def get_completion(client: OpenAI, prompt: str, model: str) -> str:
    """Get completion from OpenAI"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
