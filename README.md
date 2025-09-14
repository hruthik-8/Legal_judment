# Legal Judgment Prediction System

An AI system for predicting legal judgments in civil and criminal cases using deep learning and natural language processing with transformer models.

## 📋 Features

- **Dual-Model Architecture**: Separate models for civil and criminal cases
- **Multi-task Learning**: Simultaneous prediction of charges, applicable articles, and penalties for criminal cases
- **Binary Classification**: Yes/No prediction for civil cases
- **RESTful API**: FastAPI-based API for easy integration
- **Web Interface**: Simple HTML frontend for testing predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Scrape data from Indian Kanoon:
   ```bash
   python download_indian_kanoon.py
   ```
   This will create `data/indian_kanoon_criminal_cases.csv` and `data/indian_kanoon_civil_cases.csv`.

2. Explore the data:
   ```bash
   python explore_data.py
   ```

## 🏗️ Model Training

### Criminal Case Model (Multi-task)
```bash
python train_criminal.py \
    --epochs 3 \
    --batch 8 \
    --encoder nlpaueb/legal-bert-base-uncased \
    --lr 2e-5 \
    --max_len 512
```

### Civil Case Model (Binary Classification)
```bash
python train_civil.py \
    --epochs 3 \
    --batch 8 \
    --encoder nlpaueb/legal-bert-base-uncased \
    --lr 2e-5 \
    --max_len 384
```

## 🚀 Running the API

Start the FastAPI server:
```bash
python app.py
```
Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## 🌐 API Endpoints

### Predict Criminal Case
```http
POST /predict_criminal
Content-Type: application/json

{
    "facts": "Case facts text here...",
    "article_threshold": 0.5
}
```

Response:
```json
{
    "charge": "Charge type",
    "articles": [
        {"article": "IPC Section 123", "score": 0.8},
        ...
    ],
    "penalty_months": 12.5
}
```

### Predict Civil Case
```http
POST /predict_civil
Content-Type: application/json

{
    "facts": "Case facts...",
    "plea": "Defendant's plea",
    "law": "Relevant law context"
}
```

Response:
```json
{
    "answer": "yes",
    "probability": 0.75
}
```

## 🖥️ Web Interface

Open `static/index.html` in your browser to test the API interactively.

## 📊 Model Testing

Test the trained models:
```bash
python test_model.py
```

Or test individually:
```bash
python test_criminal.py
python test_civil.py
```

## 📈 Visualization

Generate training metric plots:
```bash
python visualize_metrics.py
```

## 📂 Project Structure

```
.
├── app.py                    # FastAPI application
├── train_criminal.py         # Criminal model training script
├── train_civil.py            # Civil model training script
├── model_multitask.py        # Multi-task model architecture
├── test_criminal.py          # Criminal model testing
├── test_civil.py             # Civil model testing
├── test_model.py             # Combined model testing
├── visualize_metrics.py      # Training metrics visualization
├── download_indian_kanoon.py # Data scraping script
├── explore_data.py           # Data exploration
├── csvdat.py                 # CSV data inspection
├── requirements.txt          # Python dependencies
├── scraper_requirements.txt  # Scraping-specific dependencies
├── static/
│   └── index.html            # Web interface
├── data/                     # Scraped datasets
├── checkpoints/              # Model checkpoints and configs
├── utils/
│   └── common.py             # Shared utilities
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
