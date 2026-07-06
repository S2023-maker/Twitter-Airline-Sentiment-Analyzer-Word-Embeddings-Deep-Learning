# ✈️ Twitter Airline Sentiment Analyzer

A Streamlit app that predicts whether an airline-related tweet is **Negative**,
**Neutral**, or **Positive**, using word-embedding features and a Keras neural
network classifier.

🔗 **Live app:** https://twitter-airline-sentiment-analyzer-word-embeddings-deep-learni.streamlit.app/

**Screenshot**: <img width="1037" height="755" alt="image" src="https://github.com/user-attachments/assets/9cf13562-3cdc-47f5-aa34-245c1432437e" />


## How it works

1. Input text is converted into a numerical vector using a pre-built word
   embedding lookup table (`embedding_lookup.pkl`).
2. The vector is scaled using a fitted `StandardScaler` (`scaler.pkl`).
3. A trained Keras model (`sentiment_model.keras`) predicts the sentiment class.
4. The predicted class index is converted back to a human-readable label using
   a `LabelEncoder` (`label_encoder.pkl`).

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit app entry point |
| `sentiment_model.keras` | Trained Keras classifier |
| `embedding_lookup.pkl` | Word → vector lookup table |
| `scaler.pkl` | Fitted `StandardScaler` |
| `label_encoder.pkl` | Fitted `LabelEncoder` for sentiment labels |
| `Airline_Sentiment.ipynb` | Original notebook used to train the model |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
git clone https://github.com/S2023-maker/Twitter-Airline-Sentiment-Analyzer-Word-Embeddings-Deep-Learning.git
cd Twitter-Airline-Sentiment-Analyzer-Word-Embeddings-Deep-Learning
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io).

**Important:** this app requires **Python 3.11** specifically, because
`tensorflow-cpu` does not currently publish wheels for newer Python versions
(3.13/3.14). If you fork or redeploy this repo:

1. Go to share.streamlit.io → **New app**
2. Select this repo, branch `main`, main file `app.py`
3. Click **Advanced settings** *before* deploying
4. Set **Python version** to **3.11**
5. Deploy

⚠️ Streamlit Cloud only applies the Python version at the moment of a fresh
deploy — rebooting an existing app or pushing new commits will **not** change
it. If you need to change the Python version later, you must delete the app
and redeploy it from scratch with the version selected again.

### requirements.txt
```
streamlit
tensorflow-cpu==2.17.0
numpy
scikit-learn
```

## Notes

- `Tweets.csv` (the original training dataset) is included for reference but
  is not required by `app.py` at runtime.
