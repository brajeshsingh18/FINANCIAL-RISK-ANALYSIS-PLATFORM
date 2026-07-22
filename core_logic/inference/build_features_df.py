from core_logic.data.get_stock_data import download_stock_data
from core_logic.data.news_data import fetch_news
from core_logic.data.macro_data import fetch_macro_data
from core_logic.preprocessing.clean_macro_data import merge_all_files
from core_logic.preprocessing.clean_stock_data import preprocess_stock_dataframe
from core_logic.preprocessing.clean_news_data import preprocess_news_data
from core_logic.features.stock_features import build_features
from core_logic.features.news_preperation import cleaning_news
from core_logic.features.generate_sentiment import feature_engineering_news
from core_logic.features.macro_features import feature_engineering_macro
from core_logic.features.final_dataset import merge_dfs
from core_logic.features.target_engineering import adding_target
from core_logic.exceptions import TickerResolutionError,DataDownloadError,FeatureEngineeringError,DataPreprocessingError,FinancialRiskPlatformError,PredictionError,InferenceDataError
from get_llm import llm
from prompts import ticker_finder_prompt
from tickers_map import company_to_ticker
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict
import joblib
import tqdm

indicators: Dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "inflation_cpi": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "gdp": "GDP",
    "treasury_10y": "DGS10",
    "treasury_3m": "TB3MS",
    "consumer_sentiment": "UMCSENT",
    "industrial_production": "INDPRO",
    "retail_sales": "RSAFS",
    "pce": "PCE",
}


def build_inference_dataframe(company:str):
    ################ our first step is to load all the data 
    try:
        ticker=llm.invoke(ticker_finder_prompt.format(company_to_ticker=company_to_ticker,user_query=company)).content.strip()  #now i am doing it with the help of llm only but it is a bit slow process to include llm while finding ticker of every company so later we will add fuzzy matyching here instead of llm call 
    except Exception as e:
        raise TickerResolutionError(
            f"Unable to resolve ticker for '{company}'.") from e
    
    try:
        stock_start_date=(datetime.now()-relativedelta(years=1)).strftime("%Y-%m-%d")
        stock_df=download_stock_data(ticker,stock_start_date)
    except Exception as e:
        raise DataDownloadError(
        f"Failed to download stock data for {ticker}.") from e
    
    try:
        news_start_time = (datetime.now() - relativedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
        news_df=fetch_news(ticker,news_start_time)     
    except Exception as e:
        raise DataDownloadError(
        f"Failed to download news data for {ticker}.") from e 
                      
    
    macro_start_time=(datetime.now()-relativedelta(years=2)).strftime("%Y-%m-%d")
    macro_dfs={}
    for indicator_name, series_id in indicators.items():
        try:
            print(f"Downloading {indicator_name} ({series_id})")
            df = fetch_macro_data(series_id, macro_start_time)

            print(
                f"{indicator_name}: "
                f"{df['Date'].min()} -> {df['Date'].max()} "
                f"rows={len(df)}"
            )

            macro_dfs[indicator_name] = df
            print("Done")
        except Exception as e:
            print(f"Failed on {indicator_name} ({series_id})")
            raise
    
    ########################### now since we have loaded all the data , we should start preprocessing all the datas one by one lets start with macro data
    try:
        for name, df in macro_dfs.items():
            print(f"{name}: {df['Date'].max()} | rows={len(df)}")
        processed_macro_data=merge_all_files(macro_dfs)
        print(processed_macro_data["Date"].max())
        print(processed_macro_data.tail())
    except Exception as e:
        raise DataPreprocessingError(
        f"Failed to preprocess macro data for {ticker}.") from e
    try:
        processed_stocks_data=preprocess_stock_dataframe(stock_df,ticker)
    except Exception as e:
        raise DataPreprocessingError(
        f"Failed to preprocess stocks data for {ticker}.") from e
    try:
        processed_news_data=preprocess_news_data(news_df,ticker)
    except Exception as e:
        raise DataPreprocessingError(
        f"Failed to preprocess news completedataframe for {ticker}.") from e
    try:
        processed_news_data=cleaning_news(processed_news_data)
    except Exception as e:
        raise DataPreprocessingError(
        f"Failed to preprocess news description text for {ticker}.") from e
    

    ######################### preprocessing is done now , we have all 3 data frames i think our job is to develop important and required features for model inference purposes
    try:
        processed_stocks_data = processed_stocks_data.set_index(["Ticker", "Date"])
        stocks_features=build_features(processed_stocks_data)
        stocks_features = stocks_features.reset_index()
        news_features=feature_engineering_news(processed_news_data)
        macro_features=feature_engineering_macro(processed_macro_data)
        # macro_feature_columns = [
        #     c for c in macro_features.columns
        #     if c != "Date"
        # ]
        # joblib.dump(
        #     macro_feature_columns,
        #     "models/macro_feature_columns.pkl"
        # )
        print("\n========== MACRO FEATURES ==========")
        print(macro_features.filter(regex="yield|inflation|fed|gdp|consumer|unemployment").tail(1).T)
    except Exception as e:
        raise FeatureEngineeringError(
        "Failed to generate technical indicators.") from e

    ####################### after adding these features, our job is to merge all 3 types of data to build single dataframe for which our model will predict output
    final_df=merge_dfs(stocks_features,news_features,macro_features)
    final_df = final_df.ffill()
    print("\n========== FINAL DF ==========")
    print(final_df.filter(regex="yield|inflation|fed|gdp|consumer|unemployment").tail(1).T)
    if final_df.empty:
        raise InferenceDataError(
            "Final inference dataframe is empty."
        )
    # final_df.drop(columns=["date","ticker"],inplace=True)   not needed here happening in classifier and regresssor files itself
    # feature_columns = joblib.load(
    #     "models/classification_feature_columns.pkl"
    # )
    # final_df=final_df.reindex(columns=feature_columns)
    latest_row=final_df.iloc[[-1]]
    print("\n========== LATEST ROW ==========")
    print(latest_row.filter(regex="yield|inflation|fed|gdp|consumer|unemployment").T)

    return {
        'final_df':final_df,
        'latest_row':latest_row,
        'ticker':ticker,
        'news':processed_news_data
    }
    
    

    


    

    






# provide_df("google")
# provide_df("googel")    
# provide_df("facebok")     
# provide_df("microsft")    
# provide_df("tesal")       
# provide_df("wallmart")    
# provide_df("jp morgan")   
# provide_df("oracle corp") 
# provide_df("adobe")       