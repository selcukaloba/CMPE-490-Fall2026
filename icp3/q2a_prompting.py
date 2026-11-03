"""Lab - Q2 (i) - Urgency by prompting

    uv run python lab/q2a_prompting.py
"""

from collections import Counter

from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:llama3.2:1b", temperature=0, num_predict=100)
complaints = open("data/complaints.md").read().split("---")

PROMPT = (
    "Rate the urgency of this customer complaint on a scale from 1 (low) "
    "to 5 (critical). Answer with the number only, no other text.\n\n"
    "Complaint:\n{complaint}"
)

counts = Counter()
for complaint in complaints:
    text = complaint.strip()
    if not text:
        continue


    reply = model.invoke(PROMPT.format(complaint=text))
    
    try:
        urgency = int(str(reply.content).strip())
    except ValueError:
        urgency = "invalid"

    if urgency not in [1, 2, 3, 4, 5]:
        urgency = "invalid"
        
    print(urgency, "|", text[:60])
    counts[urgency] += 1

print()
for urgency in [1, 2, 3, 4, 5, "invalid"]:
    print(f"urgency {urgency}: {counts[urgency]}")