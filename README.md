# Tuberculosis (TB) Screening Assistant

An AI-assisted chest radiograph pattern analysis application designed to screen frontal (PA) chest X-rays for radiographic patterns consistent with pulmonary tuberculosis.

The prototype features a restrained, clinical research interface built with Streamlit and PyTorch, running a fine-tuned DenseNet-121 binary classification model.

---

## Architecture & Technical Specifications

- **Classification Model**: DenseNet-121 (Binary Classifier)
- **Target Classes**:
  - `0`: Normal-pattern classification
  - `1`: Abnormal-pattern classification (TB-consistent pattern)
- **Input Modality**: Frontal chest radiographs (PA view) in PNG, JPEG format
- **Preprocessing Pipeline**: $224 \times 224$ px bicubic resize, tensor normalization with ImageNet parameters ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$)
- **Execution**: Local inference (CPU / CUDA automatic detection)

---

## Repository Structure

```
TB_Screening_Project/
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme and settings
├── app.py                           # Application code and inference logic
├── densenet121_combined_best.pth    # Trained DenseNet-121 checkpoint (~27.1 MB)
├── requirements.txt                 # Dependencies for deployment
├── .gitignore                       # Git ignore rules for clean repository
└── README.md                        # Documentation
```

---

## Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser to [`(https://tb-screening-assistant-cwhupnhfsttrbvmdt4l64t.streamlit.app/)`.
](https://tb-screening-assistant-cwhupnhfsttrbvmdt4l64t.streamlit.app/)
---

## Deployment to Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and select:
   - **Repository**: `<your-username>/<your-repo-name>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

*Note: The model checkpoint (`densenet121_combined_best.pth`) is ~27.1 MB, which is well below GitHub's 100 MB limit and clones directly into Streamlit Cloud without requiring Git LFS.*

---

## Research Prototype Notice & Clinical Disclaimer

This application is an artificial intelligence screening prototype developed for research, benchmarking, and investigational evaluation only.

- It is **not** an FDA-cleared, CE-marked, or clinically certified medical diagnostic device.
- It does not provide medical diagnoses, treatment plans, or clinical directives.
- Model confidence scores represent neural output calibration, not the clinical probability of active disease.
- Radiographic findings must always be reviewed by a qualified radiologist or physician in conjunction with patient history and microbiological confirmation (e.g., sputum culture, GeneXpert MTB/RIF).
