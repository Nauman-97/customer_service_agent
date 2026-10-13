import json
from pathlib import Path

from customer_agent import CustomerAgent

input_file = Path("input_email.json") #list of dicts
output_file = Path("output.json") #list of dicts
output = []


with open(input_file, "r", encoding="utf-8") as file:
    all_emails = json.load(file)

agent = CustomerAgent()

for email in all_emails:
    response = agent.get_response(email=email, retries= 2)
    output.extend(response[1])
    
    
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4)
print("hello")