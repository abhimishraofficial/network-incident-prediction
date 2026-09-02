> **Note:** This project currently uses simulated network data for development and experimentation. The architecture is designed to be extended to real-world network monitoring data.

## Problem Statement

Network incidents can cause service degradation, increased latency, packet loss, and reduced throughput. Traditional monitoring systems often detect problems after they occur.

This project explores whether historical network performance metrics and temporal patterns can be used to predict potential incidents before they occur.

The objective is to build a machine learning pipeline that can learn from network behavior and identify patterns associated with future incidents.

## Key Results

| Metric | Random Forest |
|---|---:|
| Accuracy | 99.84% |
| Precision | 43.62% |
| Recall | 95.35% |
| F1 Score | 59.85% |
| ROC-AUC | 99.95% |
| PR-AUC | 67.04% |

The model achieved a high recall of **95.35%**, meaning it successfully identified most of the incident cases in the test dataset.
## Data and Model Generation

Generated datasets and trained model files are excluded from the repository using `.gitignore`.

To reproduce the project results:

1. Generate the network data
2. Create the processed ML dataset
3. Train the machine learning models

The trained model and generated datasets will then be created locally.
## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Key Learning Outcomes

Through this project, I gained hands-on experience with:

- End-to-end Machine Learning pipeline development
- Data validation and preprocessing
- Feature engineering
- Temporal feature creation
- Lag and rolling window features
- Handling imbalanced classification problems
- Model training and evaluation
- Threshold optimization
- Precision, Recall, F1 Score, ROC-AUC, and PR-AUC
- Random Forest feature importance
- Git and GitHub project management

## Future Improvements

- Integrate real-time network monitoring data
- Add a FastAPI-based prediction API
- Build an interactive Streamlit dashboard
- Add data and model versioning
- Implement automated model retraining
- Containerize the application using Docker
- Deploy the system to a cloud platform
- Add monitoring and logging for production predictions

## Author

**Abhishek Kumar**

2025 CSE Graduate | Telecom & Network Operations | Aspiring AI/ML Engineer

---

⭐ If you found this project interesting, feel free to star the repository.