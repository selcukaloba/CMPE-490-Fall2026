"""Lab - Q2 (ii) - Urgency with a simple schema

    uv run python lab/q2b_simple_schema.py
"""

from collections import Counter

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

class Complaint(BaseModel):
    product: str
    urgency: int

model = init_chat_model("ollama:llama3.2:1b", temperature=0, num_predict=100)

structured_model = model.with_structured_output(Complaint)

complaints = open("data/complaints.md").read().split("---")

counts = Counter()
for complaint in complaints:
    text = complaint.strip()
    if not text:
        continue

    try:
        result = structured_model.invoke(text)
        urgency = result.urgency
    except Exception:
        urgency = "invalid"

    if urgency not in [1, 2, 3, 4, 5]:
        urgency = "invalid"
        
    print(urgency, "|", text[:60])
    counts[urgency] += 1

print()
for urgency in [1, 2, 3, 4, 5, "invalid"]:
    print(f"urgency {urgency}: {counts[urgency]}")