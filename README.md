# 🚀 Sentiment-Driven Stock Analysis  
**Predict NVIDIA’s next-day returns by fusing Reddit sentiment with market data.**

This project implements the framework described in “Sentiment-Driven Stock Analysis: Predicting NVIDIA’s Market Returns using Multimodal Data.” We extract and quantify retail investor mood from Reddit’s r/WallStreetBets, weight it by post virality, align it with traditional financial indicators (OHLCV + VIX), and train an LSTM to forecast NVDA returns.

---

## 🌟 Key Features

- **Reddit Sentiment Engine**  
  - Harvests NVDA-related posts from r/WallStreetBets (2021–2022).  
  - Hybrid scoring: VADER compound polarity + custom “hype” metric (emojis & catchphrases).

- **Impact Scoring**  
  - Weights each post’s sentiment by virality `(upvotes + comments)`.  
  - Aggregates to daily cumulative impact and average impact features.

- **Market Data Fusion**  
  - Retrieves NVDA OHLCV history and CBOE VIX via `yfinance`.  
  - Aligns sentiment features to next-day returns for modeling.

- **LSTM-Based Forecasting**  
  - 7-day rolling window of features (sentiment, impact, return, volume, VIX).  
  - Single LSTM layer (hidden_size=50) + dense output for regression.  
  - Trained with Adam (lr=1e-3), dropout, early stopping.

- **Visual Insights**  
  - **Line plots**: actual vs. predicted returns.  
  - **Scatter plots**: prediction vs. actual return correlation.  
  - **Timeline charts**: sentiment index and NVDA price.  
  - **Heatmaps**: feature correlations (sentiment, impact, price, VIX).

---

## 🤖 Why This Project?

Traditional quant models overlook the explosive power of retail trader sentiment. By embedding crowd mood and engagement signals into a deep‐learning pipeline, you can:

- Detect **early warning signals** when viral Reddit posts foreshadow large price moves  
- Quantify how sentiment “hype” amplifies or dampens market volatility  
- Build your own **multimodal forecasting system** for NVDA—or any ticker

---

## 🛠️ Quick Start

1. **Clone this repo** and open the main notebook (`NVDA_Sentiment_LSTM.ipynb`).  
2. **Install dependencies**:
   ```bash
   pip install pandas numpy yfinance praw vaderSentiment nltk tensorflow matplotlib seaborn
   ```
3. **Download / prepare data:**
   - Reddit CSVs from Kaggle/Pushshift or use PRAW to fetch posts.
   - No Twitter data (API costs & restrictions).
4. **Run the notebook end-to-end to:**
   - Load & preprocess Reddit + market data
   - Compute hybrid sentiment & impact scores
   - Train & evaluate the LSTM model
   - Generate visualization outputs

## 📊 Project Structure
```
.
├── data/
│   ├── wsb_2021_2022.csv      # Reddit posts dump
│   ├── NVDA_OHLCV.csv         # Historical price & volume
│   └── VIX.csv                # Market volatility index
├── notebooks/
│   └── NVDA_Sentiment_LSTM.ipynb
├── src/
│   ├── sentiment.py           # VADER + hype scoring
│   ├── impact.py              # Virality‐weighted impact features
│   ├── data_loader.py         # Data ingestion & alignment
│   └── model.py               # LSTM architecture & training script
├── outputs/
│   ├── figures/               # Line / scatter / heatmap PNGs
│   └── metrics.csv            # RMSE, R², directional accuracy
├── requirements.txt
└── README.md
```

## 💡 How It Works

**Data Collection**
- Filter WSB posts for “NVDA” mentions.
- Fetch NVDA price & VIX via yfinance.

**Feature Engineering**
- Hybrid Sentiment = VADER compound + (α × hype count)
- Impact Score = hybrid sentiment × (upvotes + comments)
- Aggregate daily mean sentiment, post volume, total impact, average impact.

**Model Training**
- Build sequences of 7-day feature windows.
- Train LSTM to minimize mean squared error on next-day return.

**Evaluation & Visualization**
- Compare to zero-return and persistence baselines.
- Plot actual vs. predicted returns, scatter, and sentiment-price timelines.

---

## 📌 Notes & Future Directions
- No Twitter data due to API costs/restrictions.
- Sarcasm & Meme Analysis: Future work will incorporate transformer-based sarcasm detection and Vision Transformers for image meme sentiment.
- Multi-Ticker Generalization: Extend framework to other equities to test robustness.

## ⚠️ Disclaimer
For educational research only. Not financial advice. Do not use for live trading without thorough backtesting.

⭐ If you find this project useful, please ⭐ the repo and drop your feedback!
