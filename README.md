# Legal Judgment Prediction System

A production-ready AI system for predicting legal judgments in both civil and criminal cases using deep learning and natural language processing.

## 📋 Features

- **Dual-Model Architecture**: Separate models for civil and criminal cases
- **Multi-task Learning**: Simultaneous prediction of charges, applicable articles, and penalties
- **High Accuracy**: State-of-the-art transformer models fine-tuned on legal texts
- **RESTful API**: Easy integration with web and mobile applications
- **Scalable**: Containerized deployment with Docker and Kubernetes support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legal-judgment-predictor.git
   cd legal-judgment-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Model Training

### Criminal Case Model (Multi-task)
```bash
python train_criminal.py \
    --epochs 5 \
    --batch 8 \
    --encoder nlpaueb/legal-bert-base-uncased \
    --learning_rate 2e-5 \
    --max_len 512
```

### Civil Case Model (Binary Classification)
```bash
python train_civil.py \
    --epochs 5 \
    --batch 8 \
    --encoder nlpaueb/legal-bert-base-uncased \
    --learning_rate 2e-5 \
    --max_len 384
```

## 🚀 Deployment

### Option 1: Local Development Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Option 2: Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t legal-judgment-predictor .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name judgment-predictor \
     -p 7860:7860 \
     -v $(pwd)/checkpoints:/app/checkpoints \
     legal-judgment-predictor
   ```

### Option 3: Kubernetes Deployment

1. Apply the Kubernetes configuration:
   ```bash
   kubectl apply -f k8s/
   ```

2. Access the service:
   ```bash
   kubectl port-forward svc/legal-judgment-predictor 7860:80
   ```

## 🌐 API Endpoints

### Predict Criminal Case
```http
POST /predict_criminal
Content-Type: application/json

{
    "facts": "[Case facts text]",
    "article_threshold": 0.5
}
```

### Predict Civil Case
```http
POST /predict_civil
Content-Type: application/json

{
    "facts": "[Case facts]",
    "plea": "[Defendant's plea]",
    "law": "[Relevant law context]"
}
```

## 📊 Model Performance

### Civil Model
- Accuracy: 82%
- F1-Score: 0.80
- ROC-AUC: 0.87

### Criminal Model
- Charge Accuracy: 75%
- Articles F1-Score: 0.70
- Penalty MAE: 4.5 months

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `checkpoints/` | Path to model checkpoints |
| `PORT` | `7860` | API server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_WORKERS` | `4` | Number of worker processes |

## 📂 Project Structure

```
.
├── app.py                # FastAPI application
├── train_civil.py        # Civil case model training
├── train_criminal.py     # Criminal case model training
├── model_multitask.py    # Multi-task model architecture
├── checkpoints/          # Model checkpoints and configs
├── static/               # Frontend assets
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
└── k8s/                  # Kubernetes manifests
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please contact [Your Name] at [your.email@example.com]
- You can swap encoders (e.g., `bert-base-uncased`) for faster debugging.
