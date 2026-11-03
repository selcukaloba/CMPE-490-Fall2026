"""Lab - Q2 (i) Bonus - Does the system prompt make a difference?

    uv run python lab/q2a_bonus_prompts.py
"""

from collections import Counter

from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:llama3.2:1b", temperature=0, num_predict=100)
complaints = [c.strip() for c in open("data/complaints.md").read().split("---") if c.strip()]

SCALE = """Urgency scale:
5 = money is being lost or is at risk right now (fraud, a stolen card, an unknown payment)
4 = the customer can't get to money they need soon
3 = a wrong charge or a mistake that needs fixing, but no harm yet
2 = a service problem or an inconvenience
1 = feedback or a suggestion, no money involved"""

PROMPTS = {
    "bare": (
        "Rate the urgency of this customer complaint on a scale from 1 (low) to 5 (critical). "
        "Answer with the number only.\n\nComplaint:\n{complaint}"
    ),
    "scale": (
        "Rate the urgency of this customer complaint.\n\n"
        + SCALE +
        "\n\nAnswer with the number only.\n\nComplaint:\n{complaint}"
    ),
    "scale+tone": (
        "Rate the urgency of this customer complaint.\n\n"
        + SCALE +
        "\n\nJudge by the risk to the customer's money, not by how angry or polite the customer sounds."
        "\nAnswer with the number only.\n\nComplaint:\n{complaint}"
    ),
}

results = {}
for name, template in PROMPTS.items():
    print(f"--- Running prompt: {name} ---")
    counts = Counter()
    for complaint in complaints:
        reply = model.invoke(template.format(complaint=complaint))
        try:
            urgency = int(str(reply.content).strip())
        except ValueError:
            urgency = "invalid"
        if urgency not in [1, 2, 3, 4, 5]:
            urgency = "invalid"
        counts[urgency] += 1
    results[name] = counts

expected = {1: 4, 2: 3, 3: 4, 4: 3, 5: 4, "invalid": 0}
print("\nurgency    expected        bare       scale  scale+tone")
for u in [1, 2, 3, 4, 5, "invalid"]:
    print(f"{u:<10} {expected[u]:>8} {results['bare'][u]:>11} {results['scale'][u]:>11} {results['scale+tone'][u]:>11}")