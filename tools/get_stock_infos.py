import os
from dotenv import load_dotenv
from urllib.request import urlopen
import certifi
import json
from tqdm.auto import tqdm

load_dotenv(verbose=True)

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def main():

    stock_infos = {}

    fmp_api_key = os.environ.get('OA_FMP_KEY')

    with open("../configs/_stock_list_/dj30.txt", "r") as op:
        stocks = op.readlines()
        stocks = [stock.strip() for stock in stocks]

    for stock in tqdm(stocks, desc="Getting stock infos", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        symbol = stock
        stock_info = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={fmp_api_key}"
        stock_info_json = get_jsonparsed_data(stock_info)
        stock_infos[symbol] = stock_info_json[0]

    with open('dj30.json', 'w') as op:
        json.dump(stock_infos, op, indent=4)


if __name__ == '__main__':
    main()