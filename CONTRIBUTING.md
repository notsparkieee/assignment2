# Contributing Guidelines

## Branch Strategy

### **Main Branch**
- Always deployable
- Only merge via Pull Requests
- All tests must pass

### **Phase Branches**
- `phase1-setup` - Project structure & configs
- `phase2-core-services` - Core business logic (chunking, embedding, Chroma)
- `phase3-search-strategies` - Search implementations (semantic, filtered, hybrid)
- `phase4-api-integration` - FastAPI routes
- `phase5-docker-deployment` - Containerization & deployment

## Workflow

1. **Start Phase:**
   ```bash
   git checkout phase1-setup
   git pull origin phase1-setup
   ```

2. **Make Changes:**
   ```bash
   # Make code changes
   git add .
   git commit -m "feat: implement chunking service"
   ```

3. **Push to GitHub:**
   ```bash
   git push origin phase1-setup
   ```

4. **Create Pull Request:**
   - Go to GitHub
   - Click "Compare & pull request"
   - Base: `main` ← Compare: `phase1-setup`
   - Add description of changes
   - Request review (if working in team)

5. **Merge to Main:**
   - After approval/review
   - Click "Merge pull request"
   - Delete phase branch (or keep for reference)

## Commit Message Convention

```
<type>: <subject>

<optional body>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code change that neither fixes nor adds feature
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples:**
```bash
git commit -m "feat: add ChromaRepository with CRUD operations"
git commit -m "fix: handle empty query in search endpoint"
git commit -m "docs: update API examples in README"
git commit -m "test: add unit tests for chunking service"
```

## Code Review Checklist

- [ ] Code follows project structure
- [ ] All new code has docstrings
- [ ] Tests added for new features
- [ ] README updated if API changed
- [ ] No hardcoded values (use config)
- [ ] Error handling implemented
- [ ] Type hints used
