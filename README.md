# 🎓 Student Performance Prediction System (ML)

An end-to-end Machine Learning web application designed to predict and analyze academic performance based on historical student study patterns, attendance, and exam scores.

---

## 🚀 Key Features
- **Predictive Modeling**: Utilizes regression/classification models trained on student behavioral and academic data.
- **Pre-trained Pipeline**: Includes pre-trained model weights (`best_student_model.pkl`) for immediate inference.
- **Interactive Interface**: Web application (`app.py`) allowing educators or students to input study metrics and evaluate performance indicators in real time.

---

## 🛠️ Tech Stack
- **Language**: Python
- **Machine Learning**: Scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Interface / Framework**: Streamlit

---

## 📁 Repository Structure
- `student_data.csv`: Source dataset of student attributes and exam metrics.
- `train_model.py`: Training script for data preprocessing, model selection, and serialization.
- `best_student_model.pkl`: Exported trained model weights.
- `dataset.py`: Utility functions for data loading and feature extraction.
- `app.py`: Interactive user dashboard for real-time predictions.

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/purnenduchhotaray-bot/StudentPerformance-ML.git](https://github.com/purnenduchhotaray-bot/StudentPerformance-ML.git)
cd StudentPerformance-ML
