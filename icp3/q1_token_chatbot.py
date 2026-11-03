"""Lab - Q1 - A chatbot that counts its tokens ..."""

from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:llama3.2:1b")

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

    print(f"Assistant: {reply.content}")
    print(
        f"  [this turn: {turn_in} in, {turn_out} out | "
        f"total: {total_in} in, {total_out} out, {total_in + total_out} tokens]"
    )