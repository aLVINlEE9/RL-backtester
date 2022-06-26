from os import EX_CANTCREAT
import pandas as pd
import os
SYSTEM_PATH = os.getcwd()

class Environment:
    def __init__(self, train_value):
        self.top = train_value[0]
        self.end_d = train_value[1]
        self.recent_d = train_value[2]


    def load_data(self, code_np, n):
        try:
            minute_db = pd.read_csv(f"{SYSTEM_PATH}/traindata_files/"
                        f"top[{self.top}]recent[{self.recent_d}]end_d[{self.end_d}]/"
                        f"data[{str(code_np[n][1])}_{str(code_np[n][0])}].csv")
            return minute_db, len(minute_db) - 5

        except Exception as e:
            print(f"{e} : data[{str(code_np[n][1])}_{str(code_np[n][0])}].csv")
            
            return -1, -1

    def load_test_data(self, code_np, n):
        try:
            minute_db = pd.read_csv(f"{SYSTEM_PATH}/testdata_files/"
                            f"Random recent[{self.recent_d}]end_d[{self.end_d}]/"
                            f"data[{str(code_np[n][1])}_{str(code_np[n][0])}].csv")
            return minute_db, len(minute_db) - 5

        except Exception as e:
            print(f"{e} : data[{str(code_np[n][1])}_{str(code_np[n][0])}].csv")
            
            return -1, -1    


    def observe(self, minute_db, n):
        try:
            if len(minute_db) - 5 == n:
                n = n - 1
            code, date, time, price, data = [], [], [], [], []
            for i in range(6):
                minute_list = minute_db.loc[n + i].tolist()
                code.append(str(minute_list[1]).zfill(6))
                date.append(minute_list[2])
                time.append(minute_list[3])
                price.append(minute_list[4])
                data.append(minute_list[5:])
            # print([n, code, date, time, price, data])
            return [n, code, date, time, price, data]
            
                
        except Exception as e:
            # self.reset()
            print(f"exception{e} : {n}")

