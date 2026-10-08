# Git Commands

| Category | Command | What It Does |
|---|---|---|
| **Setup** | `git config --global user.name "Name"` | Set your name for commits |
| **Setup** | `git config --global user.email "you@example.com"` | Set your email for commits |
| **Setup** | `git config --list` | View all Git configuration settings |
| **Setup** | `git init` | Initialize a new Git repository in the current folder |
| **Setup** | `git clone <url>` | Download (copy) an existing remote repo to your machine |
| **Staging & Committing** | `git status` | Show changed/staged/untracked files |
| **Staging & Committing** | `git add <file>` | Stage a specific file |
| **Staging & Committing** | `git add .` | Stage all changed files in current directory |
| **Staging & Committing** | `git add -p` | Stage changes interactively, chunk by chunk |
| **Staging & Committing** | `git commit -m "message"` | Save staged changes as a new commit |
| **Staging & Committing** | `git commit -am "message"` | Stage all tracked files + commit in one step |
| **Staging & Committing** | `git commit --amend` | Edit the most recent commit (message or content) |
| **History** | `git log` | Show commit history |
| **History** | `git log --oneline` | Compact one-line-per-commit history |
| **History** | `git log --oneline --graph --all` | Visual branch/commit graph |
| **History** | `git show <commit>` | Show details/diff of a specific commit |
| **History** | `git blame <file>` | Show who last changed each line of a file |
| **Branching** | `git branch` | List local branches |
| **Branching** | `git branch -a` | List all branches (local + remote) |
| **Branching** | `git branch <name>` | Create a new branch |
| **Branching** | `git branch -d <name>` | Delete a branch (safe, only if merged) |
| **Branching** | `git branch -D <name>` | Force-delete a branch (even if unmerged) |
| **Branching** | `git branch -m <new-name>` | Rename current branch |
| **Branching** | `git checkout <branch>` | Switch to a branch |
| **Branching** | `git checkout -b <branch>` | Create a new branch and switch to it |
| **Branching** | `git switch <branch>` | Modern command to switch branches |
| **Branching** | `git switch -c <branch>` | Modern command to create + switch to a branch |
| **Merging & Rebasing** | `git merge <branch>` | Merge `<branch>` into the current branch |
| **Merging & Rebasing** | `git merge --abort` | Cancel a merge that has conflicts |
| **Merging & Rebasing** | `git rebase <branch>` | Replay current branch's commits on top of `<branch>` |
| **Merging & Rebasing** | `git rebase -i HEAD~n` | Interactively edit/squash/reorder last n commits |
| **Merging & Rebasing** | `git rebase --abort` | Cancel an in-progress rebase |
| **Merging & Rebasing** | `git cherry-pick <commit>` | Apply a specific commit from another branch |
| **Remotes** | `git remote -v` | List remote repositories and their URLs |
| **Remotes** | `git remote add origin <url>` | Link local repo to a remote (e.g. GitHub) |
| **Remotes** | `git remote remove <name>` | Remove a remote link |
| **Remotes** | `git push origin <branch>` | Upload local commits to the remote branch |
| **Remotes** | `git push -u origin <branch>` | Push + set upstream tracking (first push) |
| **Remotes** | `git push --force` | Force-overwrite remote history (use with caution) |
| **Remotes** | `git push --force-with-lease` | Safer force-push; fails if remote has new commits |
| **Remotes** | `git pull origin <branch>` | Fetch + merge remote changes into current branch |
| **Remotes** | `git pull --rebase` | Fetch + rebase instead of merge |
| **Remotes** | `git fetch origin` | Download remote changes without merging |
| **Comparing** | `git diff` | Show unstaged changes vs last commit |
| **Comparing** | `git diff --staged` | Show staged changes vs last commit |
| **Comparing** | `git diff <branch1> <branch2>` | Compare two branches |
| **Undoing** | `git restore <file>` | Discard unstaged changes in a file |
| **Undoing** | `git restore --staged <file>` | Unstage a file (keeps the changes) |
| **Undoing** | `git reset --soft HEAD~1` | Undo last commit, keep changes staged |
| **Undoing** | `git reset --mixed HEAD~1` | Undo last commit, keep changes unstaged (default) |
| **Undoing** | `git reset --hard HEAD~1` | Undo last commit and discard all changes (destructive) |
| **Undoing** | `git revert <commit>` | Create a new commit that undoes a given commit (safe for shared branches) |
| **Undoing** | `git clean -fd` | Remove untracked files and folders |
| **Stashing** | `git stash` | Temporarily shelve uncommitted changes |
| **Stashing** | `git stash list` | List all stashes |
| **Stashing** | `git stash pop` | Reapply the most recent stash and remove it from the list |
| **Stashing** | `git stash apply` | Reapply a stash but keep it in the list |
| **Stashing** | `git stash drop` | Delete a stash without applying it |
| **Tags** | `git tag` | List tags |
| **Tags** | `git tag <name>` | Create a lightweight tag on current commit |
| **Tags** | `git tag -a <name> -m "msg"` | Create an annotated tag with a message |
| **Tags** | `git push origin <tag>` | Push a specific tag to remote |
| **Tags** | `git push origin --tags` | Push all tags to remote |
| **Inspecting Files** | `git ls-files` | List all tracked files |
| **Inspecting Files** | `git show HEAD:<file>` | View a file's content at the last commit |
