import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
SYSTEM_PATH = os.getcwd()

class Visualizer:
    def __init__(self):
        pass

    def conv(self, date, time):
        date = str(date)
        time = time[-5:]
        return pd.to_datetime(f'{date} {time}')

    def data_set(self, minute_db):
        self.date_1 = minute_db['date'][4]
        self.date = minute_db['date'][5]
        self.code = minute_db['code'][5]
        self.plt_minute_db = minute_db.copy()
        self.plt_minute_db = self.plt_minute_db.drop(self.plt_minute_db.index[[0, 1, 2, 3, 4]])
        self.plt_minute_db.reset_index(drop=True, inplace=True)
        self.plt_minute_db['time'] = [self.conv(self.date, str(self.plt_minute_db['time'][n])) for n in range(len(self.plt_minute_db['time']))]
        self.buy_list, self.sell_list, self.profit_list = [], [], []
        self.profit_sum_stock = 0

    
    def throw_action(self, action, profit):
        self.profit_sum_stock += profit
        self.profit_list.append(self.profit_sum_stock)
        if action == 0 or action == 3:
            self.buy_list.append(True)
        else:
            self.buy_list.append(False)
        if action == 1 or action == 4:
            self.sell_list.append(True)
        else:
            self.sell_list.append(False)

        
    def plot_graph(self, n, str):
        self.plt_minute_db['Buy'] = self.buy_list
        self.plt_minute_db['Sell'] = self.sell_list
        self.plt_minute_db['Profit'] = self.profit_list
        plt.figure(n)
        self.plt_minute_db['Buy_ind'] = np.where( (self.plt_minute_db['Buy'] > self.plt_minute_db['Buy'].shift(1)), 1, 0)
        self.plt_minute_db['Sell_ind'] = np.where( (self.plt_minute_db['Sell'] > self.plt_minute_db['Sell'].shift(1)), 1, 0)

        #print("buy", self.plt_minute_db['Buy'])
        #print("sell", self.plt_minute_db['Sell'])
        fig, ax1 = plt.subplots()
        ax1.plot(self.plt_minute_db['time'], self.plt_minute_db['close'],linewidth=0.5,color='black')

        plt.scatter(self.plt_minute_db.loc[self.plt_minute_db['Buy_ind'] ==1 , 'time'].values, \
                    self.plt_minute_db.loc[self.plt_minute_db['Buy_ind'] ==1, 'close'].values, \
                    label='skitscat', color='red', s=25, marker="^")
        plt.scatter(self.plt_minute_db.loc[self.plt_minute_db['Sell_ind'] ==1 , 'time'].values,\
                    self.plt_minute_db.loc[self.plt_minute_db['Sell_ind'] ==1, 'close'].values, \
                    label='skitscat', color='blue', s=25, marker="v")

        ax2 = ax1.twinx()
        ax2.plot(self.plt_minute_db['time'], self.plt_minute_db['Profit'], color='deeppink')

        ## Adding labels
        #plt.xlim(self.date_1, self.date)
        plt.xlabel('Time')  
        plt.ylabel('Close Price')  
        plt.title(f'{self.code} {str}rl buy and sell signal') 

        # Saving image
        plt.savefig(f'{SYSTEM_PATH}/img/{str}[{self.date} {self.code}] rl Buy sell signal.png')
        plt.close()
