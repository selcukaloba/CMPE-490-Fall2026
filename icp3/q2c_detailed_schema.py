"""Lab - Q2 (iii) - Urgency with proper fields and definitions

    uv run python lab/q2c_detailed_schema.py
"""

from collections import Counter
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

SCALE = """Urgency scale:
5 = money is being lost or is at risk right now (fraud, a stolen card, an unknown payment)
4 = the customer can't get to money they need soon
3 = a wrong charge or a mistake that needs fixing, but no harm yet
2 = a service problem or an inconvenience
1 = feedback or a suggestion, no money involved"""

class Complaint(BaseModel):
    product: Literal["credit_card", "debit_card", "loan", "savings_account", "mobile_app", "other"] = Field(
        description="The product the complaint is about"
    )
    urgency: int = Field(
        ge=1, 
        le=5, 
        description="Urgency from 1 (low) to 5 (critical) based ONLY on the risk to the customer's money"
    )

model = init_chat_model("ollama:llama3.2:1b", temperature=0, num_predict=100)

structured_model = model.with_structured_output(Complaint)

complaints = open("data/complaints.md").read().split("---")

SYSTEM_PROMPT = (
    "You are a bank assistant. Rate the urgency of the customer complaint.\n\n"
    f"{SCALE}\n\n"
    "IMPORTANT: Judge by the risk to the customer's money, not by how angry or polite the customer sounds."
)

counts = Counter()
for complaint in complaints:
    text = complaint.strip()
    if not text:
        continue

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    try:
        result = structured_model.invoke(messages)
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