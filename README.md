# Sleep_Quality_Predictor
Predict sleep quality (Good or Poor) using a machine learning model built with Streamlit and Random Forest based on lifestyle factors like sleep hours, screen time, caffeine, and stress.

🧠 Overview
The Sleep Quality Predictor is a machine learning web application that predicts whether a person’s sleep quality is Good or Poor based on various lifestyle factors such as sleep duration, screen time, caffeine intake, exercise, and stress levels.

This project demonstrates a complete ML pipeline — from synthetic data generation and model training to web deployment using Streamlit.

🌙 Key Features
<pre>

✅ Predicts Good or Poor sleep quality
✅ Built using Random Forest Classifier for high accuracy
✅ Includes a Streamlit web app for real-time predictions
✅ Synthetic dataset generation for reproducibility
✅ Displays model confidence (probability of good sleep)
</pre>

⚙️ Tech Stack
<pre>

Programming Language:	Python
Libraries:	Pandas, NumPy, Scikit-Learn, Joblib
Visualization:	Matplotlib
Web Framework:	Streamlit
Dataset:	Generated synthetically using NumPy
</pre>

📂 Project Structure
<pre>
Sleep_Quality_Predictor/
│
├── app/
│   └── streamlit_app.py          # Streamlit user interface
│
├── data/
│   └── sleep_data.csv            # Generated dataset
│
├── models/
│   └── sleep_model.joblib        # Trained ML model
│
├── results/
│   ├── metrics.txt               # Model evaluation metrics
│   └── roc_curve.png             # ROC curve visualization
│
├── scripts/
│   ├── generate_dataset.py       # Creates synthetic dataset
│   ├── train_model.py            # Trains Random Forest model
│   └── predict_example.py        # Checks prediction consistency
│
├── requirements.txt
└── README.md
</pre>
🚀 How to Run Locally
<pre>

1️⃣ Clone the repository
git clone https://github.com/sujalvaghasiya12/Sleep_Quality_Predictor.git
cd Sleep_Quality_Predictor

2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Generate dataset & train model
python scripts/generate_dataset.py
python scripts/train_model.py

5️⃣ Run the Streamlit web app
streamlit run app/streamlit_app.py


Open the URL printed in your terminal (usually http://localhost:8501).
</pre>


