"""Lab - Q1 Bonus - Switch to a smaller model when the chat gets long

Start with llama3.2:1b. When the cumulative token total passes 300,
switch to qwen2.5:0.5b for the rest of the chat, keeping the history.

    uv run python lab/q1_bonus_switch_model.py
"""

from langchain.chat_models import init_chat_model

BIG_MODEL = "ollama:llama3.2:1b"
SMALL_MODEL = "ollama:qwen2.5:0.5b"
THRESHOLD = 300 
current_model_name = BIG_MODEL
model = init_chat_model(current_model_name)

messages = [{"role": "system", "content": "You are a helpful assistant. Be brief."}]
total_in = total_out = 0

while True:
    user = input("You: ").strip()
    if user.lower() == "quit":
        break

    messages.append({"role": "user", "content": user})
    reply = model.invoke(messages)
    messages.append({"role": "assistant", "content": reply.content})

    usage = reply.usage_metadata or {}
    turn_in = usage.get("input_tokens", 0)
    turn_out = usage.get("output_tokens", 0)
    total_in += turn_in
    total_out += turn_out
    total = total_in + total_out

    print(f"Assistant: {reply.content}")
    print(
        f"  [{current_model_name} | this turn: {turn_in} in, {turn_out} out | "
        f"total: {total_in} in, {total_out} out, {total} tokens]"
    )

    if current_model_name == BIG_MODEL and total > THRESHOLD:
        print(
            f"  >>> {total} tokens used, over the limit of {THRESHOLD}: "
            f"switching to {SMALL_MODEL}. The conversation so far is kept."
        )
        current_model_name = SMALL_MODEL
        model = init_chat_model(SMALL_MODEL)