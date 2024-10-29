import requests

url = "http://localhost:8000/generate_code/"
prompt = "Write a Python function to calculate the factorial of a number"

response = requests.post(url, json={"prompt": prompt})

if response.status_code == 200:
    generated_code = response.json().get("generated_code")
    print("Generated Code:")
    print(generated_code)
else:
    print("Error:", response.status_code, response.text)
