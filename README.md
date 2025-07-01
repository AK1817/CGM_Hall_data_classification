# 🧠 Glucose Feature Extraction & Classification Pipeline

This project processes clinical, CGM (Continuous Glucose Monitoring), and lab data to classify individuals into:
- ✅ Healthy
- ⚠️ Prediabetic
- ❌ Diabetic

It includes:
- Feature extraction from glucose time series
- Classification and clustering
- Model evaluation
- End-to-end containerization and CI/CD pipeline with GitLab



## 🗂️ Project Structure

dzd_app/
│
├── hall.py # Main Python script (converted from Jupyter)
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
├── .gitlab-ci.yml # GitLab CI/CD pipeline
├── README.md # You're here!
└── Hall/ # Data files (a1c.db, ogtt_2hours_FBG.tsv, cgm.s010)



## 🚀 Running Locally with Docker

### 🐳 Build Docker image:

```bash
docker build -t dzd-app .
▶️ Run the container:
bash
Copy
Edit
docker run dzd-app
⚙️ GitLab CI/CD Pipeline
This project includes a GitLab CI/CD pipeline that:

Builds the Docker image

Pushes it to the GitLab Container Registry

Configure Secrets:
In GitLab repo → Settings > CI/CD > Variables, add:

Key	Value (example)
CI_REGISTRY	registry.gitlab.com
CI_REGISTRY_USER	Your GitLab username
CI_REGISTRY_PASSWORD	Your personal access token (with scopes)

🧪 Features Extracted
From CGM data:

TIR, TOR, PIR, MGE, MAGE, LBGI, HBGI, ADRR, MODD, CONGA24

Signal-based: frequency, mobility, Shannon entropy, etc.

📊 Models Used
Random Forest (with hyperparameter tuning)

Logistic Regression

SVM

RFE (feature selection)

KMeans (clustering + adjusted Rand index)

🧰 Requirements
Install with:

bash
Copy
Edit
pip install -r requirements.txt
Key libraries:

pandas, numpy, scikit-learn, imbalanced-learn

antropy, scipy, matplotlib

📬 Output
The pipeline prints evaluation metrics to console and can be extended to:

Save CSVs

Export plots

Upload results to cloud storage (optional)

🧑‍💻 Author
Arpit, July 2025
Open-source for research & reproducibility. ✨




