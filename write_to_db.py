import pandas as pd
import numpy as np
import csv
from pymysql import create_engine

DATA_INP = '/Users/ashwins/repo/DataScience/dailymed_crf/data/'
DATA_OP = '/Users/ashwins/repo/DataScience/dailymed_crf/inference/'

def create_conn():
    eng = create_engine('mysql+pymysql://sherlock:z00mrxr0cks!@69.164.196.100:3306/metathesaurus',echo=True)
    return eng

def get_existing(eng):
    sql = "SELECT * FROM DD_map_consolidated where dailymed = 0;"
    df = pd.read_sql(sql, eng)
    return df

def get_new():
    df = pd.read_csv(DATA_OP + 'mapped.csv', encoding='utf-8')
    return df

def main():
    eng = create_conn()
    df_e = get_existing(eng)
    df_n = get_new()
    df_e['disease'] = df_e['disease'].str.lower()
    df_n['disease'] = df_n['disease'].str.lower()

    df = pd.merge(df_n, df_e, left_on='disease', right_on='disease', how = 'left')
    df['dailymed'] = 1
    df.fillna(subset = ['metathesaurus'], value=0, inplace=True)
    df_e = df_e.loc[~df_e['disease'].isin(df['disease'].values)]
    df_final = pd.concat([df, df_e], ignore_index=True)

if __name__ == '__main__':
    main()