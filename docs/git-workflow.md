# Git & GitHub Collaboration Workflow — RAIN-REPAIR X

**Status:** Approved  
**Version:** 1.0.0  
**Target:** Monorepo collaboration across 7 sub-teams.

---

## 1. Branch Strategy

```
[main] (Protected: Production-ready baseline, tagged SIH competition releases)
  ^
  | Pull Request (Requires Tech Lead sign-off & CI green)
[development] (Protected: Team integration branch)
  ^
  | Pull Request (Requires code review & CI pass)
  +--- feature/data-pipeline        (Team A)
  +--- feature/regime-classifier    (Team B)
  +--- feature/quantile-model       (Team B)
  +--- feature/backend-api          (Team C)
  +--- feature/frontend-dashboard   (Team D)
  +--- feature/district-gis         (Team E)
  +--- feature/devops-ci            (Team F)
  +--- feature/testing-suite        (Team G)
```

---

## 2. Step-by-Step Developer Workflow

### Step 1: Sync with Latest Development
```bash
git checkout development
git pull origin development
```

### Step 2: Create a Scoped Feature Branch
```bash
git checkout -b feature/data-nwp-ingest
```

### Step 3: Implement, Test & Verify Locally
```bash
# Run linters and tests before committing
pytest tests/
```

### Step 4: Commit with Conventional Commits
```bash
git add src/data/ingest.py tests/unit/test_data_ingest.py
git commit -m "feat(data): implement NetCDF parser for raw GFS 24h precipitation"
```

### Step 5: Push & Open a GitHub Pull Request
```bash
git push -u origin feature/data-nwp-ingest
```
Open a PR against the `development` branch on GitHub. Fill out the PR template:
- Summary of changes
- Contracts validated
- Tests executed locally
- Verification screenshot/log if applicable

---

## 3. Pull Request & Review Rules

1. **Mandatory Reviews:** At least one peer review + Senior Tech Lead approval before merging into `development`.
2. **Squash and Merge:** Feature branches should be squash-merged into `development` to keep commit history clean.
3. **Never push directly to `main` or `development`:** All changes must pass through a PR.
4. **CI Checks Must Pass:**
   - Code formatting (`ruff check .`, `black --check .`)
   - Unit & contract tests (`pytest tests/`)
