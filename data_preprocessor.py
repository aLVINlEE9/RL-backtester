from typing import ValuesView
from numpy.lib.function_base import append
from numpy.lib.shape_base import expand_dims
from pandas.core import accessor
from db_parser import MarketDB, MinuteDB
from datetime import date, datetime, time, timedelta
import os
SYSTEM_PATH = os.getcwd()
import pandas as pd

class Data_Generater:
    def __init__(self, train_value):
        self.mar = MarketDB()
        self.min = MinuteDB()
        self.prep = DataPrep()
        self.top = train_value[0]
        self.end_d = train_value[1]
        self.recent_d = train_value[2]
        if self.top != -1:
            os.mkdir(f"{SYSTEM_PATH}/traindata_files/"
                    f"top[{self.top}]recent[{self.recent_d}]end_d[{self.end_d}]")
        elif self.top == -1:
            os.mkdir(f"{SYSTEM_PATH}/testdata_files/"
                    f"Random recent[{self.recent_d}]end_d[{self.end_d}]")            
        
    def valid_date(self, date):
        """return 에서 하나를 빼면됨"""
        d = date
        flag = 0
        while True:
            d -= timedelta(days=1)
            df = self.mar.get_ma_code_db(d)
            if df.empty:
                flag = 1
                continue
            else:
                break
        if flag == 0:
            d += timedelta(days=1)
        return d

    def get_random_value_code(self):
        done = True
        cnt = 0
        end_d_cnt = self.end_d
        code_df = pd.DataFrame()

        while done:
            df = self.mar.get_ma_code_db(end_d_cnt)
            if df.empty:
                end_d_cnt -= timedelta(days=1)
                continue
            df = df[['code', 'date', '거래대금', '시가총액']]
            df = df.astype({'code' : 'str'})
            df = df.sample(n = 10)
            df.reset_index(drop=True, inplace=True)
            code_df = code_df.append(df, ignore_index=True)
            end_d_cnt -= timedelta(days=1)
            cnt += 1
            if cnt >= self.recent_d:
                done = False
        code_df.to_csv(f"{SYSTEM_PATH}/testdata_files/"
                        f"Random recent[{self.recent_d}]end_d[{self.end_d}]/"
                        f"Random_recent{self.recent_d}days_{self.end_d}].csv")
        return code_df


    def get_top_value_code(self):
        """함수 설명 : 거래대금 상위 종목으로 학습 데이터 리스트를 csv 파일로 변환함

        top(int) : 거래대금 상위 몇인지에 대한 파라미터
        end_d(date) : 마지막 검색 기준 날짜
        recent_d(int) : 최근 몇일동안 검색할지"""
        
        done = True
        cnt = 0
        end_d_cnt = self.end_d
        code_df = pd.DataFrame()

        while done:
            df = self.mar.get_ma_code_db(end_d_cnt)
            if df.empty:
                end_d_cnt -= timedelta(days=1)
                continue
            df = df[['code', 'date', '거래대금', '시가총액']]
            df = df.astype({'code' : 'str'})
            df = df.sort_values(by='거래대금', ascending=False)
            df = df.head(self.top)
            df.reset_index(drop=True, inplace=True)
            code_df = code_df.append(df, ignore_index=True)
            #code_df.reset_index(drop=True, inplace=True)
            end_d_cnt -= timedelta(days=1)
            cnt += 1
            if cnt >= self.recent_d:
                done = False
        code_df.to_csv(f"{SYSTEM_PATH}/traindata_files/"
                        f"top[{self.top}]recent[{self.recent_d}]end_d[{self.end_d}]/"
                        f"code[top{self.top}_recent{self.recent_d}days_{self.end_d}].csv")
        return code_df


    def get_top_value_db(self, train_code_df):
        train_code_np = train_code_df[['code', 'date']].to_numpy()
        for n in range(len(train_code_np)):
            df_1 = self.min.get_minute_db(str(train_code_np[n][0]), str(train_code_np[n][1]))
            df_2 = self.min.get_minute_db(str(train_code_np[n][0]), str(self.valid_date\
                (train_code_np[n][1] - timedelta(days=1))))
            df = self.prep.data_prep(df_1, df_2, train_code_np[n][1], self.valid_date\
                (train_code_np[n][1] - timedelta(days=1)))
            if not df_1.empty and not df_2.empty:
                df.to_csv(f"{SYSTEM_PATH}/traindata_files/"
                            f"top[{self.top}]recent[{self.recent_d}]end_d[{self.end_d}]/"
                            f"data[{str(train_code_np[n][1])}_{str(train_code_np[n][0])}].csv")
            else:
                continue

        return train_code_np

    def get_random_value_db(self, test_code_df):
        test_code_np = test_code_df[['code', 'date']].to_numpy()
        for n in range(len(test_code_np)):
            df_1 = self.min.get_minute_db(str(test_code_np[n][0]), str(test_code_np[n][1]))
            df_2 = self.min.get_minute_db(str(test_code_np[n][0]), str(self.valid_date\
                (test_code_np[n][1] - timedelta(days=1))))
            df = self.prep.data_prep(df_1, df_2, test_code_np[n][1], self.valid_date\
                (test_code_np[n][1] - timedelta(days=1)))
            if not df_1.empty and not df_2.empty:
                df.to_csv(f"{SYSTEM_PATH}/testdata_files/"
                            f"Random recent[{self.recent_d}]end_d[{self.end_d}]/"
                            f"data[{str(test_code_np[n][1])}_{str(test_code_np[n][0])}].csv")
            else:
                continue

        return test_code_np       
        
        

class DataPrep:
    def __init__(self):
        pass


    def data_prep(self, df_1, df_2, date, date_1):
        try:
            prep_df = pd.DataFrame()
            """
            open / last close
            high / close
            low / close
            close / last close
            volume / last volume
            close / MA5close
            close / MA10 close
            close / MA20 close
            close / MA60 close
            close / MA120 close
            volume / MA5 volume
            volume / MA10 volume
            volume / MA20 volume
            volume / MA60 volume
            volume / MA120 volume
            acc_buy / acc_sell * 100 (1min)
            MA3 acc_ratio (3min)
            """
            df = df_2.append(df_1)
            prep_df[['code', 'date', 'time', 'close']] = df[['code', 'date', 'time', 'close']]

            tmp_df = df
            tmp_df['last_close'] = tmp_df['close'].shift(1)

            prep_df['open / last close'] = tmp_df[['open', 'last_close']].apply(lambda x: self.calc_ratio(x.open, x.last_close), axis=1)
            prep_df['high / close'] = df[['high', 'close']].apply(lambda x: self.calc_ratio(x.high, x.close), axis=1)
            prep_df['low / close'] = df[['low', 'close']].apply(lambda x: self.calc_ratio(x.low, x.close), axis=1)
            
            tmp_df = df
            tmp_df['last_close'] = tmp_df['close'].shift(1)
            prep_df['close / last close'] = tmp_df[['close', 'last_close']].apply(lambda x: self.calc_ratio(x.close, x.last_close), axis=1)
            
            tmp_df = df
            tmp_df['last_pvolume'] = tmp_df['pvolume'].shift(1)
            prep_df['pvolume / last pvolume'] = tmp_df[['pvolume', 'last_pvolume']].apply(lambda x: self.calc_ratio(x.pvolume, x.last_pvolume), axis=1)

            prep_df['close / MA5close'] = self.get_ma(df['close'], 5)
            prep_df['close / MA10close'] = self.get_ma(df['close'], 10)
            prep_df['close / MA20close'] = self.get_ma(df['close'], 20)
            prep_df['close / MA60close'] = self.get_ma(df['close'], 60)
            prep_df['close / MA120close'] = self.get_ma(df['close'], 120)
            prep_df['pvolume / MA5pvolume'] = self.get_ma(df['pvolume'], 5)
            prep_df['pvolume / MA10pvolume'] = self.get_ma(df['pvolume'], 10)
            prep_df['pvolume / MA20pvolume'] = self.get_ma(df['pvolume'], 20)
            prep_df['pvolume / MA60pvolume'] = self.get_ma(df['pvolume'], 60)
            prep_df['pvolume / MA120pvolume'] = self.get_ma(df['pvolume'], 120)
            prep_df['acc_ratio'] = df[['acc_buy', 'acc_sell']].apply(lambda x: self.calc_ratio(x.acc_buy, x.acc_sell) * 100, axis=1)
            prep_df['MA3acc_ratio'] = prep_df['acc_ratio'].rolling(window=3).mean()

            #print(prep_df)
            under_df = prep_df[prep_df['date'] == date_1]
            under_df = under_df.tail(5)
            prep_df = prep_df[prep_df['date'] == date]
            prep_df = under_df.append(prep_df, ignore_index=True)
            prep_df.reset_index(drop=True, inplace=True)
            return prep_df
        except Exception as e:
            print(f"{e}")

    def get_ma(self, pram, win):
       return pram / pram.rolling(window=win).mean()

    def calc_ratio(self, x, y):
        if x == 0 or y == 0:
            return 0
        return x / y

#a = Data_Generater()
#print(a.min.get_minute_db('005930', date(2021, 10,22)))
