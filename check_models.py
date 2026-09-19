# check_models.py  — run once: python check_models.py
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()
models = client.models.list()
for m in sorted(models.data, key=lambda x: x.id):
    print(m.id)