from datetime import date, datetime, timedelta
from data_preprocessor import Data_Generater
# from environmnet import Environment
from db_parser import *
import numpy as np
import random
import pandas as pd
from environmnet import Environment
from agent import Agent
from visualizer import Visualizer
import pylab
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import os
SYSTEM_PATH = os.getcwd()

if __name__ == "__main__":
    train_value = [10, date(2021, 11, 26), 3] # 상위 10종목 date로부터 최근 10일간 code
    # epoch[0], state_size[1], action_size[2], trade money[3], discounting_factor[4], learning_rate[5], epsilon[6], 
    # epsilon_decay[7], epsilon_min[8], batch_size[9], train_start[10], max_memory[11]
    hyper_parameter = [100, 19, 5, 1000000, 0.99, 0.001, 1.0, \
                        0.999, 0.01, 64, 1000, 2000]
    epoch = hyper_parameter[0]
    state_size = hyper_parameter[1]
    action_size = hyper_parameter[2]
    trade_money = hyper_parameter[3]
    discounting_factor = hyper_parameter[4]
    learning_rate = hyper_parameter[5]
    epsilon = hyper_parameter[6]
    epsilon_decay = hyper_parameter[7]
    epsilon_min = hyper_parameter[8]
    batch_size = hyper_parameter[9]
    train_start = hyper_parameter[10]
    max_memory = hyper_parameter[11]

    def build_model():
        model = Sequential()
        model.add(Dense(100, input_shape=(6, 19), activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(100, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(100, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    model = build_model()
    model.load_weights("./stock_dqn.h5")

    

    env = Environment(train_value)
    agent = Agent(hyper_parameter, env)
    visual = Visualizer()

    # datagn = Data_Generater(train_value)

    # test_code_df = datagn.get_top_value_code() # 그 목록
    # code_np = datagn.get_top_value_db(test_code_df) # 해당 조건의 minute data를 전처리해서 각각 csv로 변환


    # test
    profit_sum = 0
    profit_list, stock_n_list = [], []

    try:  
        code_df = pd.read_csv(f"{SYSTEM_PATH}/traindata_files/"
                            f"top[{train_value[0]}]recent[{train_value[2]}]end_d[{train_value[1]}]/"
                            f"code[top{train_value[0]}_recent{train_value[2]}days_{train_value[1]}].csv")
        code_df['code'] = code_df['code'].apply(lambda x : str(x).zfill(6))
        code_np = code_df[['code', 'date']].to_numpy()

        profit_sum = 0
        profit_list, stock_n_list = [], []

                # 각 종목별로 observation 하기 (epoch)
        for stock_n in range(len(code_np)):

            try:
                # data 로드
                minute_db, length = env.load_data(code_np, stock_n)
                
                # print(minute_db)
                # agent 정보 reset
                agent.reset_agent()
                # 첫 state 설정
                state = agent.get_state(env.observe(minute_db, 0), -1, 0)
                # print(state)
                # sample들을 임시 저장할 리스트 생성
                sample_list = []

                visual.data_set(minute_db)
                buy_hold_cnt = 0
                sell_hold_cnt = 0
                action_point = -1
                profit = 0
            # 한 종목 별로
                for n in range(length):
                    # 해당 시간(분)에 대한 정보
                    observation = env.observe(minute_db, n) # [idx, code, date, time, price, data]
                    # print(f"observe{observation[3]}--------------------------={n}")

                    if action_point == 0: # buy~
                        buy_hold_cnt += 1
                        sell_hold_cnt = 0
                    elif action_point == 1 or action_point == -1: # sell~ 
                        buy_hold_cnt = 0
                        sell_hold_cnt += 1   

                    # action (epsilon-greedy policy로 불러옴)
                    # action_point(지금 action이 after buy[1]인지 after sell[1]인지 확인)
                    action, action_point = agent.get_action_test(state, buy_hold_cnt, sell_hold_cnt, model)

                    #print(f"get_action{action} {action_point}--------------------------={n}")

                    # agent가 해당 행동을 하였을때 대한 reward 반환
                    immediate_reward, profit, action = agent.act(action, buy_hold_cnt, sell_hold_cnt, observation) # im_re : 즉시 보상, dly_re : 지연 보상
                    #print(f"act{immediate_reward} {profit}--------------------------={n}")
                    
                    visual.throw_action(action, profit)
 

                    # next state를 가져옴
                    if length == n + 1:
                        if action == 1:
                            next_state = agent.get_state(env.observe(minute_db, n), action_point, immediate_reward)
                    else:
                        next_state = agent.get_state(env.observe(minute_db, n + 1), action_point, immediate_reward)
                    # print(f"get_state{state[5][16]} {next_state[5][16]}--------------------------={n}")
                    profit_sum += profit

                    state = next_state
                    #print("stock:", stock_n, "n:", n, "reward:", immediate_reward)
                
                # 한 종목을 
                profit_list.append(profit_sum)
                stock_n_list.append(stock_n)
                print("stock:", stock_n, "  profit_sum:", profit_sum, "  epsilon:", epsilon)

                visual.plot_graph(stock_n, "test")

            except Exception as e:
                print(f"Exception occured : {e} ({stock_n})")


            pylab.figure(len(code_np))
            pylab.plot(stock_n_list, profit_list, 'b')
            pylab.savefig("./test.png")

    except Exception as e:
        print(f"Exception occured: {e}")    

    #test_code = prep.get_top_value_code(10, date(2021, 11, 12), 2) # 상위 10종목 date로부터 최근 2일간(train * 0.2)
