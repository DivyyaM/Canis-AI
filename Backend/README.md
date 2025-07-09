# Canis Backend - AutoML Platform

Hey! I'm Divya, and I built this AutoML backend to make machine learning easier for teams. It's not just another ML tool—it's a complete platform that handles everything from data upload to model deployment.

## What It Does

**AutoML Made Simple**
- Upload your data (CSV, Excel, etc.) and it automatically figures out what you're trying to predict
- Tests 15+ different ML models to find the best one for your data
- Explains why models make certain predictions (no more black boxes!)
- Deploys models so you can use them right away

**Built for Teams**
- Different user roles (admin, data scientist, business user, etc.)
- Everyone gets the right level of access
- Workflows that multiple people can use together
- Keeps track of all your models and experiments

**Production Ready**
- Runs in Docker containers (easy to deploy anywhere)
- Handles multiple users and heavy workloads
- Monitors everything and alerts you when something goes wrong
- Built-in security and backup systems

## Quick Start

**For Development:**
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
uvicorn app.main:app --reload
```

**For Production:**
```bash
docker-compose up -d
```

**First Time Setup:**
```bash
# Create admin account
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -d "username=admin&email=admin@company.com&password=admin123&role=admin"

# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -d "username=admin&password=admin123"
```

## How It Works

1. **Upload Data** → The system analyzes your data and suggests what to predict
2. **Train Models** → Tests multiple algorithms and picks the best one
3. **Get Explanations** → Understand why the model makes certain predictions
4. **Deploy & Use** → Put the model into production with one click

## Key Features

- **Smart Data Analysis**: Automatically detects what you're trying to predict
- **Model Comparison**: Tests multiple algorithms side-by-side
- **Explainability**: SHAP and LIME explanations for every prediction
- **Team Management**: Role-based access control
- **Workflow Automation**: Pre-built pipelines for common tasks
- **Monitoring**: Health checks and performance tracking

## API Examples

```bash
# Upload data
POST /api/v1/upload-csv/

# Train a model
POST /api/v1/train-model/

# Get predictions
POST /api/v1/predict/

# Explain a prediction
POST /api/v1/explain-model/
```

## Why I Built This

I wanted something that:
- Makes ML accessible to everyone, not just experts
- Gives you confidence in your models (no black boxes)
- Works in real companies with real teams
- Handles the boring stuff so you can focus on insights

## Getting Help

- Check out the interactive docs at `http://localhost:8000/docs`
- All the code is well-commented and easy to extend
- Built with standard Python tools (FastAPI, scikit-learn, etc.)

---

*Built for real-world ML workflows, not just demos.* 