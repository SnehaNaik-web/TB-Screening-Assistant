# Tuberculosis (TB) Screening Assistant

An AI-assisted chest X-ray screening application designed to identify normal and abnormal radiographic patterns that may be consistent with pulmonary tuberculosis.

The prototype features a clean, clinical interface built using Streamlit and PyTorch, powered by a fine-tuned DenseNet-121 binary classification model. It aims to make AI-assisted screening support more accessible, particularly in areas with limited internet connectivity.

## 🚀 Live Demo

**Try the TB Screening Assistant online:**

[Open TB Screening Assistant](https://tb-screening-assistant-cwhupnhfsttrbvmdt4l64t.streamlit.app/)

The application allows users to upload a frontal chest X-ray and view the model's predicted class and confidence score.

**Important:** This is an experimental research prototype, not a clinically validated diagnostic tool. Its predictions do not confirm or rule out tuberculosis.

---

## Architecture & Technical Specifications

* **Classification Model:** DenseNet-121 binary classifier
* **Target Classes:**

  * `0`: Normal-pattern classification
  * `1`: Abnormal-pattern classification (TB-consistent pattern)
* **Input Modality:** Frontal chest radiographs (PA view)
* **Accepted Formats:** PNG, JPG, JPEG
* **Preprocessing:** Resize to 224 × 224 pixels using bicubic interpolation, followed by ImageNet normalization:

  * Mean: `[0.485, 0.456, 0.406]`
  * Standard deviation: `[0.229, 0.224, 0.225]`
* **Inference Framework:** PyTorch
* **Execution:** CPU inference on the hosted application; local execution may use CPU or CUDA, depending on availability.

---

## Repository Structure

```text
TB_Screening_Project/
├── .streamlit/
│   └── config.toml
├── app.py
├── densenet121_combined_best.pth
├── TB_Screening_Model.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

* `app.py` — Streamlit application and model inference logic.
* `densenet121_combined_best.pth` — Trained DenseNet-121 model checkpoint.
* `TB_Screening_Model.ipynb` — Model development and evaluation notebook.
* `requirements.txt` — Python dependencies.
* `.streamlit/config.toml` — Streamlit interface configuration.
* `.gitignore` — Files excluded from version control.

---

## Local Setup & Execution

### 1. Clone the Repository

Replace the repository URL below with your GitHub repository URL.

```bash
git clone https://github.com/SnehaNaik-web/TB-Screening-Assistant.git
cd TB-Screening-Assistant
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

Keep the terminal running while using the local application.

---

## Deployment to Streamlit Community Cloud

This project is deployed using Streamlit Community Cloud.

To deploy your own instance:

1. Push the project repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**.
4. Choose the GitHub repository.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

The trained model checkpoint is approximately 27.1 MB, below GitHub's 100 MB individual file limit.

**Live application:**
https://tb-screening-assistant-cwhupnhfsttrbvmdt4l64t.streamlit.app/

---

## Dataset & Model Evaluation

The model was developed using publicly available chest X-ray datasets:

* Shenzhen chest X-ray dataset
* Montgomery County chest X-ray dataset

The combined dataset contains 800 images, divided into training, validation, and test subsets.

The preliminary held-out test split produced the following results:

| Metric         | Result |
| -------------- | -----: |
| Accuracy       | 84.17% |
| Macro F1-score | 0.8417 |
| ROC-AUC        | 0.9341 |

These results reflect evaluation on the project's test split. They should not be interpreted as evidence of clinical performance or independent external validation.

---

## Research Prototype Notice & Clinical Disclaimer

This application is an experimental artificial intelligence screening prototype developed for research, benchmarking, and investigational evaluation.

* It identifies normal and abnormal patterns in chest X-ray images.
* An abnormal-pattern prediction does not confirm tuberculosis.
* A normal-pattern prediction does not reliably rule out tuberculosis.
* The displayed confidence score represents the model's confidence in its predicted class, not the probability that a patient has TB.
* The application is not FDA-cleared, CE-marked, or clinically certified as a medical diagnostic device.
* It must not be used as the sole basis for diagnosis, treatment decisions, or clinical directives.

The project's goal is to explore how AI-assisted image screening could support more accessible preliminary screening workflows, especially in low-connectivity settings.

---

## Developed For

**DevHack 2026 — Problem Statement 3.1**

**Institution:** NMAM Institute of Technology, Nitte

**GitHub Repository:**
https://github.com/SnehaNaik-web/TB-Screening-Assistant

**Live Demo:**
https://tb-screening-assistant-cwhupnhfsttrbvmdt4l64t.streamlit.app/

