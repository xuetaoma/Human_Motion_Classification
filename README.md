# Motion History Image (MHI) Based Human Action Classification

**Author:** Xuetao Ma  

## Overview

This project implements a lightweight human action recognition pipeline using **Motion History Images (MHI)** and **Hu moments** as features, followed by classification using **Random Forest**. The goal is to recognize six human actions from the KTH dataset: `boxing`, `handclapping`, `handwaving`, `jogging`, `running`, and `walking`.

Unlike deep learning approaches, this project uses interpretable, handcrafted features and traditional machine learning, achieving decent performance while remaining easy to understand and fast to compute.

Dataset is from: https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/

---

## Folder Structure
.
├── Visualization.ipynb
├── best_params.txt
├── config.py
├── config_optimal.py
├── final_predict.py
├── final_prediction_set
├── mhi_utils.py
├── models
├── predict_sample.py
├── readme.md
├── requirements.txt
├── run.py
├── sample_set
├── train_set
│   ├── boxing
│   ├── handclapping
│   ├── handwaving
│   ├── jogging
│   ├── running
│   └── walking
├── train_set

## How It Works

1. **Motion History Image (MHI)** is computed from frame differencing with a binary threshold θ and decay constant τ.
2. From each MHI, **7 Hu moments** are computed to extract translation, scale, and rotation-invariant motion features.
3. A **Random Forest classifier** is trained on these features for classification.
4. Models are trained across a grid of θ and τ values.
5. A separate prediction set (`final_prediction_set/`) is used to evaluate model accuracy.

---

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the script
- Place **6 labeled videos** under `sample_set/` to test prediction output.
- Place **~10 or more videos** under `final_prediction_set/` for accuracy evaluation.

### Step 3: Train & Evaluate

- Open `run.py` and set your desired range of `Tau` and `Threshold`.
- Then run:

```bash
python run.py
```
This will train and save models in `/models/` and log results to `best_params.txt`.

### Step 4: Visualize
- Open `Visualization.ipynb` in Jupyter
- Run each cell to visualize accuracy heatmaps and identify the best-performing model configurations.

### Extra notes:
If you would like to check performance of one model, change the parameters under config_optimal.py
Then run:
```bash
python final_predict.py
```
