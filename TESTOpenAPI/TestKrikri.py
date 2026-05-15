from openai import OpenAI

api_key = "token-abc123"
base_url = "http://localhost:8000/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

system_prompt = "Είσαι ένα ανεπτυγμένο μεταφραστικό σύστημα που απαντάει με λίστες Python. Δεν γράφεις τίποτα άλλο στις απαντήσεις σου πέρα από τις μεταφρασμένες λίστες."
user_prompt = "Δώσε μου την παρακάτω λίστα με μεταφρασμένο κάθε string της στα ελληνικά: ['Ethics of duty', 'Postmodern ethics', 'Consequentialist ethics', 'Utilitarian ethics', 'Deontological ethics', 'Virtue ethics', 'Relativist ethics']"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

response = client.chat.completions.create(model="ilsp/Llama-Krikri-8B-Instruct",
                                          messages=messages,
                                          temperature=0.0,
                                          top_p=0.95,
                                          max_tokens=8192,
                                          stream=False)

print(response.choices[0].message.content)
# ['Ηθική καθήκοντος', 'Μεταμοντέρνα ηθική', 'Συνεπειοκρατική ηθική', 'Ωφελιμιστική ηθική', 'Δεοντολογική ηθική', 'Ηθική αρετών', 'Σχετικιστική ηθική']
