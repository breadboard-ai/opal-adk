"""Contains agent instructions for the generate functions."""

GENERATE_TEXT_FUNCTION = "generate_text"

GENERATE_INSTRUCTIONS = f"""
## When to call "{GENERATE_TEXT_FUNCTION}" function

When evaluating the objective, make sure to determine whether calling 
"{GENERATE_TEXT_FUNCTION}" is warranted. The key tradeoff here is latency: 
because it's an additional model call, the "generate_text" will take longer to 
finish.

Your job is to fulfill the objective as efficiently as possible, so weigh the
need to invoke "{GENERATE_TEXT_FUNCTION}" carefully.

Here is the rules of thumb:

- For shorter responses like a chat conversation, just do the text generation 
yourself. You are an LLM and you can do it without calling 
"{GENERATE_TEXT_FUNCTION}".
- For longer responses like generating a chapter of a book or analyzing a 
large and complex set of files, use "{GENERATE_TEXT_FUNCTION}".

"""
