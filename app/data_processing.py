# app/data_processing.py
import re
from pathlib import Path


def extract_code_files(repo_dir="repo", file_exts=[".py", ".js"]):
    code_snippets = []
    for path in Path(repo_dir).rglob("*"):
        if path.suffix in file_exts:
            with open(path, "r") as file:
                code_snippets.append(file.read())
    return code_snippets

def create_prompt_response_pairs(code_snippets):
    pairs = []
    for code in code_snippets:
        docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
        functions = re.findall(r'def (.*?):', code)
        for doc, func in zip(docstrings, functions):
            prompt = f"Write a function called {func}.\n\nDocumentation:\n{doc}"
            response = code
            pairs.append({"prompt": prompt, "response": response})
            print(f'{"prompt": prompt, "response": response}')
    return pairs
