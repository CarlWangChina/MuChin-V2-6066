import json

concatenated_text = ""
with open('lyrics/100014.txt', 'r') as file:
    for line in file:
        if not line.strip():
            continue
        json_data = json.loads(line)
        text = json_data["text"]
        concatenated_text += text + "\n"

concatenated_text = concatenated_text.rstrip("\n")
print(concatenated_text)