# 🐕 Canis AI - Your Intelligent AutoML Companion

Welcome to **Canis AI**, an advanced AutoML backend that transforms your machine learning workflow from complex to effortless. Think of it as having a brilliant data scientist by your side, ready to handle everything from data analysis to model deployment.

## 🌟 What Makes Canis AI Special?

Canis AI isn't just another AutoML tool—it's your complete ML companion. Whether you're a seasoned data scientist or just starting your ML journey, Canis AI adapts to your needs with intelligent automation, powerful explainability features, and conversational AI guidance.

### ✨ Key Features That Set Us Apart

- **🤖 Intelligent Automation**: Upload your data and let Canis AI handle the rest
- **🧠 Conversational AI**: Chat with an AI assistant for guidance and explanations
- **📊 Advanced Explainability**: Understand your models with SHAP, LIME, and feature importance
- **🏆 Smart Benchmarking**: Automatically test multiple models and find the best one
- **📈 Real-time Monitoring**: Track model performance and health in real-time
- **🔄 Model Versioning**: Keep track of your models like a professional ML engineer

## 🚀 Quick Start

### 1. Upload Your Data
```python
import requests

# Upload from URL
response = requests.post("https://your-canis-ai-url.com/api/v1/upload-url/", 
                        params={"url": "https://example.com/your-dataset.csv"})

# Or upload a file
with open("your-data.csv", "rb") as f:
    response = requests.post("https://your-canis-ai-url.com/api/v1/upload-csv/", 
                           files={"file": f})
```

### 2. Let Canis AI Work Its Magic
```python
# Analyze your data
requests.post("https://your-canis-ai-url.com/api/v1/analyze-data/")

# Detect the target column
requests.post("https://your-canis-ai-url.com/api/v1/detect-target/")

# Get model suggestions
requests.post("https://your-canis-ai-url.com/api/v1/suggest-model/")

# Train the best model
requests.post("https://your-canis-ai-url.com/api/v1/train-model/")
```

### 3. Understand Your Results
```python
# Get model explanations
requests.get("https://your-canis-ai-url.com/api/v1/shap-explanations/")

# Chat with AI assistant
requests.post("https://your-canis-ai-url.com/api/v1/chat/", 
              params={"query": "Explain my model's performance"})
```

## 🛠️ Core Features

### 📁 Smart Data Handling
- **Multiple Formats**: CSV, Excel, TSV, JSON
- **URL Uploads**: Direct dataset imports from web sources
- **Automatic Profiling**: Understand your data at a glance
- **Target Detection**: AI automatically finds your target variable

### 🎯 Intelligent Model Selection
- **Task Classification**: Automatically determines if it's classification, regression, etc.
- **Model Suggestions**: AI recommends the best algorithms for your data
- **Hyperparameter Tuning**: Grid and random search optimization
- **Cross-Validation**: Robust performance evaluation

### 🏆 Advanced Benchmarking
- **Multi-Model Testing**: Compare 10+ algorithms automatically
- **Best Model Selection**: AI picks the winner based on performance
- **Performance Reports**: Detailed comparison and insights
- **Model Download**: Save and deploy your best models

### 🧠 Explainability & Insights
- **SHAP Explanations**: Understand feature importance globally and locally
- **LIME Explanations**: Interpret individual predictions
- **Feature Analysis**: Deep dive into what drives your model
- **Visual Reports**: Beautiful charts and graphs

### 💬 Conversational AI Assistant
- **Natural Language Queries**: Ask questions about your data and models
- **Guided Workflows**: Get step-by-step recommendations
- **Code Generation**: Automatically generate training code
- **Memory & History**: Remember your conversations and context

### 🔄 Model Management
- **Version Control**: Save, load, and manage model versions
- **Performance Tracking**: Monitor model health over time
- **Async Operations**: Handle long-running tasks efficiently
- **Download & Deploy**: Export models for production use

## 📚 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/upload-csv/` | POST | Upload dataset file |
| `/api/v1/upload-url/` | POST | Upload dataset from URL |
| `/api/v1/analyze-data/` | POST | Profile and analyze data |
| `/api/v1/detect-target/` | POST | Find target column |
| `/api/v1/suggest-model/` | POST | Get model recommendations |
| `/api/v1/train-model/` | POST | Train selected model |
| `/api/v1/benchmark-models/` | POST | Test multiple models |
| `/api/v1/chat/` | POST | Talk to AI assistant |

### Advanced Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tune-hyperparameters/` | POST | Optimize model parameters |
| `/api/v1/shap-explanations/` | GET | Get SHAP explanations |
| `/api/v1/lime-explanations/` | GET | Get LIME explanations |
| `/api/v1/download-model/` | GET | Download trained model |
| `/api/v1/save-model-version/` | POST | Save model version |
| `/api/v1/healthcheck/` | GET | Check service health |

## 🏗️ Architecture

Canis AI is built with modern, scalable technologies:

- **Backend**: FastAPI (Python) - Fast, modern, and auto-documenting
- **ML Stack**: scikit-learn, XGBoost, SHAP, LIME
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **AI Integration**: Google Gemini for conversational features
- **Deployment**: Docker + Render.com for easy scaling

## 🔧 Setup & Installation

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- Gemini API key (for AI features)

### Local Development
```bash
# Clone the repository
git clone https://github.com/DivyyaM/Canis-AI.git
cd Canis-AI/Backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the server
uvicorn app.main:app --reload
```

### Docker Deployment
```bash
# Build the image
docker build -t canis-ai .

# Run the container
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key canis-ai
```

### Render.com Deployment
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set build command: `pip install -r Backend/requirements.txt`
4. Set start command: `uvicorn Backend.app.main:app --host 0.0.0.0 --port 8000`
5. Add environment variable: `GEMINI_API_KEY=your_key`

## 🎯 Use Cases

### For Data Scientists
- **Rapid Prototyping**: Test multiple approaches quickly
- **Model Comparison**: Benchmark algorithms systematically
- **Explainability**: Generate reports for stakeholders
- **Production Ready**: Export models for deployment

### For ML Engineers
- **Automated Pipelines**: Reduce manual model selection
- **Version Control**: Track model evolution
- **Performance Monitoring**: Real-time model health checks
- **Scalable Deployment**: Containerized and cloud-ready

### For Business Users
- **No-Code ML**: Upload data and get insights
- **AI Guidance**: Chat with assistant for help
- **Visual Reports**: Beautiful, interpretable results
- **Confidence Building**: Understand model decisions

## 🔍 Troubleshooting

### Common Issues

**502 Bad Gateway on Render**
- Wait 1-2 minutes for cold start
- Check logs for startup errors
- Verify environment variables are set

**Missing Dependencies**
- Ensure all packages in `requirements.txt` are installed
- Check for version conflicts
- Verify Python version compatibility

**API Key Issues**
- Set `GEMINI_API_KEY` environment variable
- Check API key validity
- Ensure proper permissions

### Getting Help

1. **Check the logs**: Look for error messages in the application logs
2. **Test endpoints**: Use `/api/v1/healthcheck/` to verify service status
3. **Chat with AI**: Use the `/api/v1/chat/` endpoint for guidance
4. **Review docs**: Visit `/docs` for interactive API documentation

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastAPI** for the amazing web framework
- **scikit-learn** for the robust ML algorithms
- **Google Gemini** for the conversational AI capabilities
- **Render.com** for the seamless deployment platform

## 📞 Support

- **Documentation**: Visit `/docs` when the server is running
- **Issues**: Report bugs on GitHub
- **Discussions**: Join our community discussions
- **Email**: Reach out for enterprise support

---

**Ready to transform your ML workflow?** 🚀

Start with Canis AI today and experience the future of automated machine learning. Your intelligent ML companion is waiting to help you build better models, faster.

*Built with ❤️ for the ML community* 