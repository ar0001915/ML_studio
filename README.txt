ML STUDIO - SETUP GUIDE
=======================

1. Install Python 3.11 or 3.12
   Download: https://www.python.org/downloads/
   During Windows installation, tick "Add Python to PATH".

2. Extract this ZIP file to a folder, for example Desktop\ml_studio_project.

3. Open the folder in VS Code.
   VS Code: File > Open Folder > select ml_studio_project.

4. Open Terminal in VS Code (Terminal > New Terminal).

5. Create a virtual environment:
   Windows:
     py -m venv .venv
     .venv\Scripts\activate
   macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate

6. Install dependencies:
     python -m pip install --upgrade pip
     pip install -r requirements.txt

7. Run the app:
     streamlit run app.py

8. A browser tab should open (usually http://localhost:8501).
   If it doesn't, copy the URL printed in the terminal into your browser.

HOW TO USE
----------
- Upload a CSV from the left sidebar, or choose one of the synthetic sample datasets.
- Dataset Overview: inspect rows, column types, missing values and statistics.
- Data Cleaning: select missing-value handling, remove duplicates, download cleaned CSV.
- EDA: view distributions, scatterplots and correlation heatmap.
- Train Models: select the target column, choose Regression/Classification or Auto-detect,
  choose algorithms, and press Train selected models.
- Predict: after training, enter values and generate a prediction.

NOTES
-----
- This is a student project prototype. Auto-detection is a heuristic; confirm the task type.
- For classification, target values should represent classes (e.g. Yes/No, category names).
- For regression, target should be numeric (e.g. price, score, sales).
- Use CSV files with headers. Very large files may require more memory/time.
