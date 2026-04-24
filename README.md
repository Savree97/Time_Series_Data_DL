# ⚡ Electricity Consumption Forecasting — Deep Learning Models

> Comparing **LSTM**, **GRU**, **SimpleRNN**, and **Vision Transformer** on daily electricity consumption time series using a multivariate approach with engineered temporal features.

---

## 📌 Project Overview

This project implements and benchmarks four deep learning architectures on the task of **next-day electricity consumption prediction**. Rather than a simple univariate approach, the models are trained on a **multivariate feature matrix** that includes engineered time features alongside raw consumption values — capturing weekly, monthly, and seasonal patterns that pure value-based models miss.

---

## 📂 Dataset

**Source:** [Predict Electricity Consumption — Kaggle](https://www.kaggle.com/code/nageshsingh/predict-electricity-consumption)

| Property | Detail |
|---|---|
| File | `Energy_consumption_dataset.csv` |
| Frequency | Daily |
| Target | Electricity Consumption (MWh / units) |

> **Setup:** Download the dataset from Kaggle and place it in the project root as `Energy_consumption_dataset.csv`.

---

## 🧠 Models Implemented

| Model | Description |
|---|---|
| **SimpleRNN** | Basic recurrent network; baseline to demonstrate vanishing gradient limitations |
| **LSTM** | Long Short-Term Memory with input/forget/output gates; captures long-term dependencies |
| **GRU** | Gated Recurrent Unit; simplified LSTM with 2 gates, faster training with comparable accuracy |
| **Vision Transformer (ViT)** | Transformer-based architecture adapted for 1D time series; uses Conv1D patching, positional encoding, and multi-head self-attention |

All models use:
- 2 recurrent/transformer layers with **Dropout (0.2)**
- **Adam optimizer** (lr = 0.001)
- **MSE loss**
- **EarlyStopping** (patience = 10, restores best weights)

---

## 🔧 Feature Engineering

A **multivariate approach** is used. Each time step contains 5 features:

```
[Electricity Consumption, Month, Day of Week, Day of Month, Quarter]
```

This allows models to learn:
- **Weekly patterns** — lower consumption on weekends
- **Seasonal patterns** — higher in summer/winter
- **Monthly cycles** — billing and business effects

Studies show multivariate forecasting reduces prediction error by **15–30%** over univariate on electricity data.

---

## ⚙️ Methodology

```
Raw CSV
  → Date parsing & index setting
  → Temporal feature extraction (month, day_of_week, day_of_month, quarter)
  → Separate MinMaxScaler for features (X) and target (y)
  → Sliding window sequences (window_size = 30 days)
  → 80/20 chronological train/test split
  → Train all 4 models with EarlyStopping
  → Evaluate: RMSE, MAE, R²
  → Visualize & compare
```

---

## 📊 Evaluation Metrics

| Metric | Interpretation |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **MAE** | Mean Absolute Error — average prediction deviation |
| **R² Score** | Explained variance — closer to 1.0 is better |

---

## 📈 Visualizations

The notebook produces 4 plots:

1. **Bar chart** — RMSE, MAE, and R² comparison across all models
2. **Predictions vs Actual** — Best model's forecast plotted against ground truth
3. **Training curves** — Validation loss over epochs (log scale) for all models
4. **Feature correlation** — How each time feature correlates with electricity consumption

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 3. Add the dataset
Download from [Kaggle](https://www.kaggle.com/code/nageshsingh/predict-electricity-consumption) and place in the project root:
```
Energy_consumption_dataset.csv
```

### 4. Run the notebook
```bash
jupyter notebook DL_Assignment_Time_Series_Data.ipynb
```
Run all cells top to bottom.

---

## 🗂️ Project Structure

```
├── DL_Assignment_Time_Series_Data.ipynb   # Main notebook
├── Energy_consumption_dataset.csv         # Dataset (add manually)
└── README.md
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `tensorflow` | Model building and training |
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Scaling and evaluation metrics |
| `matplotlib` | Visualizations |

---

## 🏆 Key Findings

- **LSTM / GRU** generally outperform SimpleRNN due to gating mechanisms that prevent vanishing gradients
- **Vision Transformer** demonstrates competitive performance by leveraging attention over temporal patches
- **SimpleRNN** serves as a baseline — its lower performance quantifies the benefit of gated architectures
- Multivariate features (especially **month** and **quarter**) show meaningful correlation with consumption, validating the feature engineering approach

---

## 📝 Assignment Context

This project was developed as part of a **Deep Learning course assignment** on time series forecasting. The objective was to implement, train, and rigorously compare four architectures on a real-world energy dataset.

---

## 📄 License

This project is for educational purposes.
