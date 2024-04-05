# FinAgent

# Installation
```
conda create -n finagent python=3.10
conda activate finagent

#for linux
apt-get update && apt-get install -y libmagic-dev
# for mac
pip install python-magic-bin==0.4.14

conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install -r requirements.txt
playwright install
# for linux
playwright install-deps
```

# Prepare the environment variables
The suggested way to do it is to create a .env file in the root of the repository (never push this file to GitHub) where variables can be defined.
Please check the examples below.
Sample .env file containing private info that should never be on git/GitHub:
```
OA_OPENAI_KEY = "abc123abc123abc123abc123abc123" # https://platform.openai.com/docs/overview
OA_FMP_KEY = "abc123abc123abc123abc123abc123" # https://site.financialmodelingprep.com/developer/docs
OA_POLYGON_KEY = "abc123abc123abc123abc123abc123" # https://polygon.io/
OA_YAHOOFINANCE_KEY = "abc123abc123abc123abc123abc123" # https://finnhub.io/
HUGGINEFACE_KEY = "abc123abc123abc123abc123abc123" # https://huggingface.co/
RAPIDAPI_KEY = "abc123abc123abc123abc123abc123" # https://rapidapi.com/
ALPHA_VANTAGE_KEY = "abc123abc123abc123abc123abc123" # https://www.alphavantage.co/
```

# Prepare the data
```

1. Download the data
python tools/download_prices.py
python tools/download_news.py
...

2. Process the data
python tools/data_process.py
```

# Run
```
python main.py
python main_mi_w_decision.py
...
```