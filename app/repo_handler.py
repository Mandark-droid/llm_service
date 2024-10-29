# # app/repo_handler.py
# from GitPython import Repo
# from pathlib import Path
#
#
# def clone_repo(git_url, dest_dir="repo"):
#     if not Path(dest_dir).exists():
#         Repo.clone_from(git_url, dest_dir)
#     else:
#         print("Repository already cloned.")
#
# def pull_repo(dest_dir="repo"):
#     repo = Repo(dest_dir)
#     repo.remotes.origin.pull()
#     print("Repository updated.")


# app/repo_handler.py
import os
import subprocess


def clone_repos(git_urls):
    for git_url in git_urls:
        repo_name = git_url.split('/')[-1].replace('.git', '')
        clone_dir = os.path.join("repos", repo_name)  # Define your clone directory

        if not os.path.exists(clone_dir):
            print(f"Cloning {git_url} into {clone_dir}...")
            try:
                subprocess.run(['git', 'clone', git_url, clone_dir], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error cloning {git_url}: {e}")
        else:
            print(f"Repository {repo_name} already exists. Pulling the latest changes...")
            try:
                subprocess.run(['git', '-C', clone_dir, 'pull'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error pulling from {git_url}: {e}")

def pull_repo(clone_dir):
    """Pulls the latest changes from the repository in the specified directory."""
    try:
        subprocess.run(["git", "-C", clone_dir, "pull"], check=True)
        print(f"Pulled latest changes in {clone_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling repository: {e}")
