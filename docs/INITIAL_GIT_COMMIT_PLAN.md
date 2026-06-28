# Initial Git Commit Plan

Recommended first commits:

1. `chore(repo): add safe gitignore and repository structure`
2. `docs(workflow): add GitHub research workflow`
3. `test(repo): add lightweight reproducibility checks`
4. `ci(repo): add repository validation workflow`
5. `results(registry): add compact completed experiment history`

Do not commit automatically unless the working tree is clean and Git identity is explicitly configured. Stop before pushing.

Before the first commit:

```powershell
git status --short
python -m unittest discover -s tests
python outputs\experiment_registry\update_registry.py --validate
```

Confirm no datasets, embeddings, fitted models, participant predictions, credentials, or large generated artifacts are staged.
