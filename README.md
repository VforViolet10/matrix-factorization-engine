# Matrix Factorization Engine

### 🚀 [Matrix Factorization Engine](https://matrix-factorization-engine-601x8yb4s-violet26.vercel.app/)

### 📚 [API Documentation](https://matrix-factorization-engine-1.onrender.com/docs)

</p>

---

## 📌 Overview

**Matrix Factorization Engine** is a research-oriented Python framework and full-stack web platform for implementing, analyzing, comparing, and applying matrix and tensor factorization techniques.

The project combines mathematical algorithms with practical applications in:

* Linear algebra
* Numerical computing
* Machine learning
* Recommendation systems
* Image compression
* Dimensionality reduction
* Performance benchmarking

The platform provides both a **Python-based computational engine** and an **interactive web application** powered by FastAPI and Next.js.

---

## ✨ Features

### 🧮 Matrix & Tensor Factorization

The engine implements multiple factorization and decomposition techniques through a unified architecture.

### 🎬 Recommendation System

A matrix-factorization-based recommendation application supporting:

* SVD
* NMF
* Configurable latent components
* User selection
* Predicted ratings
* Top-K recommendations

### 🖼️ Image Compression

Low-rank SVD is used to compress images while measuring:

* Compression ratio
* Reconstruction error
* Relative reconstruction error
* RMSE

### 📊 Recommendation Evaluation

The platform evaluates recommendation predictions using:

* RMSE
* Precision@K
* Recall@K
* NDCG@K

### ⚡ Benchmarking & Scalability

Experiments measure:

* Runtime
* Memory usage
* Reconstruction quality
* Compression efficiency
* Scalability across matrix sizes

### 🌐 Interactive Web Platform

The Next.js frontend provides four main modules:

1. **Factorization**
2. **Recommendations**
3. **Image Compression**
4. **Evaluation**

Users can interact with the engine without writing Python code.

---

# 🧮 Algorithms

## Classical Decompositions

| Algorithm | Description                  |
| --------- | ---------------------------- |
| **SVD**   | Singular Value Decomposition |
| **QR**    | QR Factorization             |
| **LU**    | LU Decomposition             |
| **Eigen** | Eigenvalue Decomposition     |

## Advanced Methods

| Algorithm                | Description                        |
| ------------------------ | ---------------------------------- |
| **NMF**                  | Non-negative Matrix Factorization  |
| **Incremental SVD**      | Incremental low-rank factorization |
| **Robust PCA**           | Low-rank and sparse decomposition  |
| **Tensor Factorization** | CP/PARAFAC decomposition using ALS |

---

# 🏗️ Architecture

```text
                         ┌──────────────────────────────┐
                         │       Next.js Frontend       │
                         │            Vercel            │
                         │                              │
                         │  Factorization               │
                         │  Recommendations             │
                         │  Image Compression           │
                         │  Evaluation                  │
                         └──────────────┬───────────────┘
                                        │
                                        │ REST API
                                        ▼
                         ┌──────────────────────────────┐
                         │       FastAPI Backend        │
                         │            Render             │
                         │                              │
                         │  /methods                    │
                         │  /factorize                  │
                         │  /applications/...           │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │   Matrix Factorization       │
                         │          Engine               │
                         │                              │
                         │ SVD • NMF • QR • LU          │
                         │ Eigen • Incremental SVD      │
                         │ Robust PCA • Tensor ALS      │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │        Applications          │
                         │                              │
                         │ Image Compression            │
                         │ Recommendation Systems       │
                         │ Recommendation Evaluation    │
                         │ Benchmarking                 │
                         └──────────────────────────────┘
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
├── api/
│   ├── main.py
│   └── routes/
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

# 🔌 REST API

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

Interactive Swagger documentation is available through the **API Documentation** link at the top of this README.

---

# 🖼️ Image Compression

The image compression application uses low-rank SVD to approximate images using a reduced number of singular components.

It reports:

* Original image dimensions
* Compressed dimensions
* Number of components
* Compression ratio
* Reconstruction error
* Relative reconstruction error
* RMSE

### Example Results

| Components | Compression Ratio | Relative Error |
| ---------: | ----------------: | -------------: |
|          5 |            23.95× |       0.004472 |
|         10 |            11.98× |       0.004468 |
|         20 |             5.99× |       0.004458 |
|         40 |             2.99× |       0.004432 |
|         80 |             1.50× |       0.004412 |

These experiments demonstrate the trade-off between the number of retained components, compression ratio, and reconstruction quality.

---

# 🎬 Recommendation System

The recommendation application demonstrates collaborative filtering using a movie-rating matrix.

The system supports:

* SVD-based recommendations
* NMF-based recommendations
* Configurable latent dimensions
* User selection
* Configurable recommendation count
* Predicted ratings

Example dataset:

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

The evaluation module uses a held-out test dataset to evaluate recommendation predictions.

### Metrics

**RMSE**

Measures the difference between predicted and actual ratings.

**Precision@K**

Measures the proportion of recommended items within the top-K results that are relevant.

**Recall@K**

Measures the proportion of relevant items retrieved within the top-K results.

**NDCG@K**

Measures ranking quality while giving greater importance to relevant items appearing higher in the recommendation list.

The interface allows users to configure:

* K
* Relevance threshold
* Training ratings
* Test ratings
* Predictions

---

# 📈 Benchmarking

The project contains experiments for analyzing algorithm performance across different matrix sizes and configurations.

Benchmarking focuses on:

* Execution time
* Memory consumption
* Reconstruction error
* Relative reconstruction error
* Compression ratio
* Scalability

The benchmark infrastructure allows factorization methods to be evaluated quantitatively across different workloads.

---

# 🧪 Testing

The project includes an automated test suite covering the mathematical engine, applications, metrics, and API functionality.

Current test status:

```text
189 passed
2 warnings
```

Run the tests with:

```bash
pytest
```

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/VforViolet10/matrix-factorization-engine.git
cd matrix-factorization-engine
```

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
source venv/Scripts/activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
pip install .
```

## Run Tests

```bash
pytest
```

## Run the API

```bash
uvicorn api.main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

---

# 💻 Frontend Development

The frontend is located in the `frontend/` directory.

```bash
cd frontend
npm install
npm run dev
```

The development application will be available at:

```text
http://localhost:3000
```

For local backend development, configure:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

For the deployed application, the frontend connects to the deployed FastAPI backend.

---

# ☁️ Deployment

The application uses a separated frontend/backend architecture:

```text
Frontend
   │
   │ Vercel
   ▼
Next.js Application
   │
   │ REST API
   ▼
FastAPI Backend
   │
   │ Render
   ▼
Matrix Factorization Engine
```

The deployed frontend and API documentation are linked at the top of this README.

---

# 🛠️ Technology Stack

### Backend & Mathematical Engine

* Python
* NumPy
* SciPy
* scikit-learn
* Pandas
* Matplotlib
* FastAPI
* Uvicorn
* Pytest

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

### Development & Deployment

* Git
* GitHub
* GitHub Actions
* Vercel
* Render

---

# 🎯 Project Objectives

The project explores the intersection of:

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

The main objective is to connect the mathematical foundations of matrix factorization with practical, measurable applications.

---

# 🔬 Research & Engineering Focus

The engine is designed around evaluating factorization methods using multiple dimensions:

```text
                         Matrix / Tensor
                                │
                                ▼
                         Factorization
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
             Accuracy         Runtime         Memory
                │               │               │
                └───────────────┼───────────────┘
                                ▼
                          Reconstruction
                                │
                                ▼
                           Applications
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
           Compression   Recommendations   Machine Learning
```

This provides a foundation for studying both the mathematical behavior and practical performance of factorization algorithms.

---

# 🔮 Future Extensions

Potential future work includes:

* GPU acceleration
* Distributed factorization
* Larger recommendation datasets
* Interactive benchmark visualization
* Tensor-factorization demonstrations
* Additional dimensionality-reduction methods
* Experiment tracking
* Advanced model comparison
* Persistent experiment results

---

# 👩‍💻 Author

**Bushra Farhad**

Data Science • AI/ML • Linear Algebra • Technology

GitHub: [VforViolet10](https://github.com/VforViolet10)

---

# 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.
