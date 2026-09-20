# Matrix Factorization Engine

<p align="center">

**A research-oriented matrix factorization platform combining linear algebra, machine learning, real-world applications, benchmarking, and an interactive web interface.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=next.js\&logoColor=white)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/Tests-189%20passed-brightgreen)](#testing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</p>

<p align="center">

### 🚀 [Live Demo](https://matrix-factorization-engine-601x8yb4s-violet26.vercel.app/)

### 📚 [API Documentation](https://matrix-factorization-engine-1.onrender.com/docs)

</p>

---

## Overview

**Matrix Factorization Engine** is a research-oriented Python framework and interactive web platform for implementing, comparing, and applying matrix and tensor factorization techniques.

The project combines:

* Classical linear algebra decompositions
* Advanced matrix factorization algorithms
* Tensor factorization
* Low-rank approximation
* Image compression
* Recommendation systems
* Recommendation evaluation
* Performance benchmarking
* A REST API built with FastAPI
* An interactive Next.js frontend

The goal is to provide a unified environment for studying how different factorization techniques behave in terms of reconstruction, approximation quality, computational performance, and real-world applications.

---

## 🌐 Live Application

### Interactive Web Platform

**Frontend:**
https://matrix-factorization-engine-601x8yb4s-violet26.vercel.app/

The web application provides four interactive modules:

1. **Factorization**
2. **Recommendation System**
3. **Image Compression**
4. **Recommendation Evaluation**

### Backend API

**FastAPI Backend:**
https://matrix-factorization-engine-1.onrender.com/

**Interactive API Documentation:**
https://matrix-factorization-engine-1.onrender.com/docs

The frontend communicates with the deployed FastAPI backend through REST endpoints.

---

## ✨ Features

### Matrix Factorization

The engine supports multiple decomposition and factorization techniques through a unified interface.

### Real-World Applications

The project demonstrates factorization techniques through practical applications:

* Image compression using low-rank SVD
* Collaborative-filtering-style recommendation systems
* Recommendation prediction evaluation
* Dimensionality reduction
* Low-rank approximation

### Benchmarking

The project includes experiments for analyzing:

* Reconstruction error
* Relative reconstruction error
* Runtime
* Memory usage
* Compression ratio
* Scalability
* Approximation quality

### Interactive Web Interface

The Next.js frontend allows users to interact with the engine without writing Python code.

Users can:

* Enter custom matrices
* Select factorization algorithms
* Configure the number of components
* Generate recommendations
* Upload images for compression
* Evaluate recommendation predictions
* Inspect numerical results

---

# 🧮 Algorithms

## Classical Matrix Decompositions

| Algorithm | Description                  |
| --------- | ---------------------------- |
| **SVD**   | Singular Value Decomposition |
| **QR**    | QR Factorization             |
| **LU**    | LU Decomposition             |
| **Eigen** | Eigenvalue Decomposition     |

## Advanced Factorization Methods

| Algorithm                | Description                                    |
| ------------------------ | ---------------------------------------------- |
| **NMF**                  | Non-negative Matrix Factorization              |
| **Incremental SVD**      | Incremental / streaming low-rank factorization |
| **Robust PCA**           | Low-rank + sparse decomposition                |
| **Tensor Factorization** | CP/PARAFAC decomposition using ALS             |

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────────────┐
                         │       Next.js Frontend      │
                         │          Vercel             │
                         │                             │
                         │  • Factorization            │
                         │  • Recommendations          │
                         │  • Image Compression        │
                         │  • Evaluation               │
                         └──────────────┬──────────────┘
                                        │
                                        │ REST API
                                        ▼
                         ┌─────────────────────────────┐
                         │       FastAPI Backend       │
                         │           Render            │
                         │                             │
                         │  /factorize                 │
                         │  /methods                   │
                         │  /applications/...          │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Matrix Factorization      │
                         │          Engine             │
                         │                             │
                         │  SVD • NMF • QR • LU        │
                         │  Eigen • Incremental SVD    │
                         │  Robust PCA • Tensor ALS    │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Applications           │
                         │                             │
                         │  Image Compression          │
                         │  Recommendation Systems     │
                         │  Evaluation                 │
                         │  Benchmarking               │
                         └─────────────────────────────┘
```

---

# 📁 Project Structure

```text
matrix-factorization-engine/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── applications/
│   ├── image_compression/
│   ├── recommendation_system/
│   └── scalability/
│
├── benchmarks/
│   ├── memory/
│   ├── performance/
│   └── scalability/
│
├── examples/
│
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── FactorizationPanel.tsx
│   │   │   ├── ImageCompression.tsx
│   │   │   ├── MatrixEditor.tsx
│   │   │   ├── RecommendationEvaluation.tsx
│   │   │   └── RecommendationSystem.tsx
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── package.json
│   └── ...
│
├── src/
│   └── matrix_factorization/
│       ├── algorithms/
│       ├── core/
│       ├── metrics/
│       └── ...
│
├── tests/
│
├── api/
│   ├── main.py
│   └── routes/
│
├── requirements.txt
├── setup.py
├── pytest.ini
├── Procfile
├── Dockerfile
├── .dockerignore
├── .python-version
├── LICENSE
└── README.md
```

---

# 🔌 API

The FastAPI backend exposes the following endpoints:

| Method | Endpoint                                 | Purpose                         |
| ------ | ---------------------------------------- | ------------------------------- |
| `GET`  | `/`                                      | API information                 |
| `GET`  | `/health`                                | Health check                    |
| `GET`  | `/methods`                               | Available factorization methods |
| `POST` | `/factorize`                             | Factorize a matrix              |
| `POST` | `/applications/recommendations`          | Generate recommendations        |
| `POST` | `/applications/recommendations/evaluate` | Evaluate predictions            |
| `POST` | `/applications/image-compression`        | Compress an uploaded image      |

### API Documentation

Interactive Swagger documentation:

https://matrix-factorization-engine-1.onrender.com/docs

---

# 💡 Example: Using the Python Engine

```python
import numpy as np

from matrix_factorization import MatrixFactorizationEngine

X = np.array([
    [5.0, 3.0, 1.0],
    [4.0, 2.0, 5.0],
    [1.0, 5.0, 4.0],
])

engine = MatrixFactorizationEngine(
    method="svd",
    n_components=2,
)

engine.fit(X)

reconstructed = engine.reconstruct()

print(reconstructed)
```

---

# 🖼️ Image Compression

The image compression application uses low-rank SVD to approximate an image using a reduced number of singular components.

The application reports:

* Original dimensions
* Compressed dimensions
* Number of components
* Compression ratio
* Reconstruction error
* Relative reconstruction error
* RMSE

Example experimental results:

| Components | Compression Ratio | Relative Error |
| ---------: | ----------------: | -------------: |
|          5 |            23.95× |       0.004472 |
|         10 |            11.98× |       0.004468 |
|         20 |             5.99× |       0.004458 |
|         40 |             2.99× |       0.004432 |
|         80 |             1.50× |       0.004412 |

These results demonstrate the trade-off between compression and reconstruction quality.

---

# 🎬 Recommendation System

The recommendation application demonstrates matrix factorization on a movie-rating matrix.

The system supports:

* SVD-based recommendations
* NMF-based recommendations
* Configurable latent components
* User selection
* Configurable recommendation count
* Predicted ratings

Example movie dataset:

```text
Inception
Interstellar
The Dark Knight
Dune
Avengers
Iron Man
The Matrix
Gladiator
```

---

# 📊 Recommendation Evaluation

The evaluation module uses a held-out test dataset to measure recommendation quality.

Metrics include:

### RMSE

Measures prediction error between predicted and actual ratings.

### Precision@K

Measures how many of the top-K recommendations are relevant.

### Recall@K

Measures how many relevant items are retrieved within the top-K recommendations.

### NDCG@K

Measures ranking quality while giving greater importance to higher-ranked relevant recommendations.

The evaluation interface allows users to configure:

* K
* Relevance threshold
* Training ratings
* Test ratings
* Predictions

---

# 📈 Benchmarking

The project includes benchmarking experiments for studying algorithmic performance across different matrix sizes and configurations.

Benchmark dimensions include:

* Runtime
* Memory consumption
* Reconstruction error
* Scalability
* Compression efficiency

This allows different factorization approaches to be compared quantitatively rather than only theoretically.

---

# 🧪 Testing

The project has an automated test suite covering the mathematical engine, applications, API, and supporting functionality.

Current test status:

```text
189 passed
2 warnings
```

Run the test suite with:

```bash
pytest
```

---

# 🚀 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/VforViolet10/matrix-factorization-engine.git

cd matrix-factorization-engine
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
source venv/Scripts/activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
pip install .
```

## 4. Run tests

```bash
pytest
```

## 5. Run the API

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 💻 Running the Frontend

Move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:3000
```

---

# ☁️ Deployment

The project uses a separated frontend/backend deployment architecture.

### Frontend

**Platform:** Vercel

https://matrix-factorization-engine-601x8yb4s-violet26.vercel.app/

### Backend

**Platform:** Render

https://matrix-factorization-engine-1.onrender.com/

### API Documentation

https://matrix-factorization-engine-1.onrender.com/docs

The frontend communicates with the backend through:

```env
NEXT_PUBLIC_API_URL=https://matrix-factorization-engine-1.onrender.com
```

---

# 🛠️ Technology Stack

## Backend / Engine

* Python
* NumPy
* SciPy
* scikit-learn
* Pandas
* Matplotlib
* FastAPI
* Uvicorn
* Pytest

## Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

## Deployment

* Vercel
* Render
* GitHub Actions

---

# 🎯 Project Objectives

The project was developed to explore the intersection of:

* Linear Algebra
* Matrix Factorization
* Numerical Computing
* Machine Learning
* Recommendation Systems
* Computer Vision
* Data Compression
* Performance Engineering
* REST API Development
* Full-Stack Data Applications

---

# 🔬 Research & Engineering Focus

The engine is designed around several measurable properties of factorization algorithms:

```text
                    Matrix / Tensor
                          │
                          ▼
                  Factorization
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       Accuracy        Runtime       Memory
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                    Applications
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
       Compression   Recommendations   ML
```

This makes the project suitable for experimentation with both mathematical properties and practical machine-learning applications.

---

# 📌 Future Extensions

Potential future work includes:

* GPU acceleration
* Distributed factorization
* Larger recommendation datasets
* Interactive benchmark visualization
* Tensor-factorization web demonstrations
* Additional dimensionality-reduction techniques
* Experiment tracking
* More advanced model comparison
* Persistent experiment results

---

# 👩‍💻 Author

**Bushra Farhad**

Data Science • AI/ML • Linear Algebra • Technology

GitHub:
https://github.com/VforViolet10

---

# 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.
