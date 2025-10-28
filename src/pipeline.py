import numpy as np
import pandas as pd
from src.utils import  sort_price, reduce_size



def merge_and_melt(sales_df, calendar_df, prices_df):
    prices_df = sort_price(prices_df)
    sales_df = reduce_size(sales_df)
    calendar_df = reduce_size(calendar_df)
    prices_df = reduce_size(prices_df)
    df = pd.melt(sales_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()
    # Merging the Calendar data into df
    df = pd.merge(df, calendar_df, on='d', how='left')
    # And now merging the Sales_price data
    df = pd.merge(df, prices_df, on=['store_id','item_id','wm_yr_wk'], how='left') 
    df = reduce_size(df)
    return df

def fill_sell(df):
    # Filling the null values in sell_price using "backward" fill
    df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].bfill()
    return df



def fill_event(df):
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.add_categories('No_Event').fillna('No_Event')
        else:
            df[col] = df[col].fillna('No_Event')
    return df


def cat_to_num(df):
    df.d = df['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,type in enumerate(types):
        if type.name == 'category':
            df[cols[i]] = df[cols[i]].cat.codes
    df.drop('date',axis=1,inplace=True)
    return df


def add_lags(df,lags):
    for lag in lags:
        df['sold_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sold'].shift(lag).astype(np.float16)
    return df


def add_mean_encoding(df):
    df['item_sold_avg'] = df.groupby('item_id')['sold'].transform('mean').astype(np.float16)
    df['state_sold_avg'] = df.groupby('state_id')['sold'].transform('mean').astype(np.float16)
    df['store_sold_avg'] = df.groupby('store_id')['sold'].transform('mean').astype(np.float16)
    df['cat_sold_avg'] = df.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
    df['dept_sold_avg'] = df.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
    df['cat_dept_sold_avg'] = df.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
    df['store_item_sold_avg'] = df.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
    df['cat_item_sold_avg'] = df.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
    df['dept_item_sold_avg'] = df.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
    df['state_store_sold_avg'] = df.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
    df['state_store_cat_sold_avg'] = df.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
    df['store_cat_dept_sold_avg'] = df.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
    return df


def add_sold_mean(df):
    df['rolling_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)
    df['expanding_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.expanding(2).mean()).astype(np.float16)
    df['daily_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sold'].transform('mean').astype(np.float16)
    df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
    df['selling_trend'] = (df['daily_avg_sold'] - df['avg_sold']).astype(np.float16)
    df.drop(['daily_avg_sold','avg_sold'],axis=1,inplace=True)
    return df

def fill_df(df):
    df = df[df['d']>30].reset_index(drop=True)
    return df