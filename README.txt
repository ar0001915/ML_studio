ML STUDIO -Intelligent data analysis & automated machine learning platform SETUP GUIDE

1. Install Python 3.11 or 3.12
   Download: https://www.python.org/downloads/
   During Windows installation, tick "Add Python to PATH".

2. Extract this ZIP file to a folder, for example Desktop\ml_studio.

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

This ML Studio is a web-based machine learning application developed using Python and Streamlit. It simplifies the machine learning workflow by providing an interactive platform for data analysis, visualization, model training, evaluation, and prediction.
 - The main goal of ML Studio is to make machine learning and data analysis easier, more accessible, and efficient through a user-friendly interface.   


