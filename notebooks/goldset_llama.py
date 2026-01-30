# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.14.2)
#     language: python
#     name: python3
# ---

# %%
import requests, json

with open('../llama_API_KEY', 'r') as f:
    key = f.read().strip()

r = requests.post("https://chat.fri.uni-lj.si/ollama/api/generate",
         headers={"Authorization": f"Bearer {key}"},
         json={"model": "llama3.3:latest",
            "stream": False,
            "prompt": "what day is it today?"})
data = json.loads(r.text)
print(data['response'])

# %% [markdown]
# test

# %%
1+1

# %%
