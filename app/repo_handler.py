# app/repo_handler.py
from git import Repo
from pathlib import Path
import os
import subprocess

def clone_repo(git_url, dest_dir="repo"):
    if not Path(dest_dir).exists():
        Repo.clone_from(git_url, dest_dir)
    else:
        print("Repository already cloned.")

def pull_repo(dest_dir="repo"):
    repo = Repo(dest_dir)
    repo.remotes.origin.pull()
    print("Repository updated.")

def clone_repos(git_urls):
    for git_url in git_urls:
        repo_name = git_url.split('/')[-1].replace('.git', '')
        clone_dir = os.path.join("repos", repo_name)  # Define your clone directory

        if not os.path.exists(clone_dir):
            Repo.clone_from(git_url, clone_dir)
        else:
            print(f"Repository {repo_name} already cloned.")
