from datetime import date
import numpy as np
import random
from itertools import chain
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

class Agent:
    def __init__(self, hyper_parameter, env):
        self.state_size = hyper_parameter[1]
        self.action_size = hyper_parameter[2]
        self.trade_money = hyper_parameter[3]
        self.epsilon = hyper_parameter[6]
        self.env = env

    # 입실론 탐욕 정책으로 행동 선택
    # 0 agg_buy -> 1 sell / 2 hold
    # 1 agg_sell -> 0 buy / 2 hold
    # 2 hold -> 0 buy / 1 sell / 2 hold
    # 3 con_buy
    # 4 con_sell

    # action_point == 1 -> sell ~
    # action_point == 0 -> buy ~

    # random = 2중 하나
    # q_val = 2중 하나

    def get_action(self, state, buy_hold_cnt, sell_hold_cnt, model):
        
        # (actions)
        # print(self.actions)
        if np.random.rand() <= self.epsilon:
            self.curr_action = self.actions[random.randrange(len(self.actions))]
            self.get_action_point(self.curr_action)
            self.actions = self.get_action_list(self.curr_action, buy_hold_cnt, sell_hold_cnt)
            return self.curr_action, self.action_point
        else:
            q_value = model.predict(state)
            #print(q_value)
            for i in range(5):
                if i in self.actions:
                    continue
                else:
                    q_value[0][i] = 0
            self.curr_action = np.argmax(q_value[0])
            self.get_action_point(self.curr_action)
            self.actions = self.get_action_list(self.curr_action, buy_hold_cnt, sell_hold_cnt)
            return self.curr_action, self.action_point

    def get_action_test(self, state, buy_hold_cnt, sell_hold_cnt, model):
        #print(state[5][16])
        #print(actions)
        q_value = model.predict(state)
        #print(q_value[0])
        for i in range(5):
            if i in self.actions:
                continue
            else:
                q_value[0][i] = 0
        #print(q_value[0])
        self.curr_action = np.argmax(q_value[0])
        self.get_action_point(self.curr_action)
        self.actions = self.get_action_list(self.curr_action, buy_hold_cnt, sell_hold_cnt)
        return self.curr_action, self.action_point



    def act(self, action, buy_hold_cnt, sell_hold_cnt, observation): # recent 5[idx, code, date, time, price, data]
        cur_price = observation[4][5]
        # print(action, buy_hold_cnt, sell_hold_cnt, self.buy_price)

        if action == 2:
            if self.action_point == 0: # hold after buy ~
                return self.calculate_ratio(self.buy_price, cur_price), 0, action
            elif self.action_point == 1 or self.action_point == -1: # hold after sell ~
                return 0, 0, action
        elif action == 0 or action == 3: # buy
            self.amount = self.bet_amount // observation[4][5]
            self.buy_price = observation[4][5]
            self.buy_amount = self.amount * observation[4][5]
            return 0, 0, action
        elif action == 1 or action == 4: # sell
            return self.calculate_ratio(self.buy_price, cur_price), (self.amount * cur_price) - self.buy_amount, action

    def calculate_ratio(self, buy_price, price):
        return ((price / buy_price) * 100) - 100

    def reset_agent(self):
        self.curr_action = -1 # reset value
        self.action_point = -1 # reset value
        self.bet_amount = self.trade_money # bet 금액
        self.buy_amount = 0 # 총 매수금액
        self.buy_price = 0 # 매수 가격
        self.amount = 0 # 매수량
        self.holds = [0, 0, 0, 0, 0, 0]
        self.rewards = [0, 0, 0, 0, 0, 0]
        self.actions = [0, 2]



    def get_state(self, observation, done, reward):
        if done == -1 or done == 1: # hold after sell
            self.holds.pop(0)
            self.holds.append(0)
        elif done == 0: # hold after buy
            self.holds.pop(0)
            self.holds.append(1)
        # 실제 reward X - > 현재 가치
        self.rewards.pop(0)
        self.rewards.append(reward)

        #time = observation[3]
        data = observation[5]
        # print(observation[1], observation[2])
        state = [[datas, [holds], [rewards]] for datas, holds, rewards in zip(data, self.holds, self.rewards)]
        state = list(chain.from_iterable(state))
        state = list(chain.from_iterable(state))
        state = np.reshape(state, [6, self.state_size])
        # np.array [list([1], hold, reward) list([2]) list([3])
        # list([4]) list([5]) list([6]) ]
        return state


    def get_action_point(self, action):
        if action == 1 or action == 4: # sell
            self.action_point = 1

        elif action == 0 or action == 3: # buy
            self.action_point = 0

    def over_3_or_not(self, cnt):
        if cnt < 4:
            return False
        elif cnt >= 4:
            return True

    def get_action_list(self, action, buy_hold_cnt, sell_hold_cnt):
        if action == 2:
            if self.action_point == 0:
                if self.over_3_or_not(buy_hold_cnt):
                    return [1, 2, 4]
                else:
                    return [1, 2]
            elif self.action_point == 1 or self.action_point == -1:
                if self.over_3_or_not(sell_hold_cnt):
                    return [0, 2, 3]
                else:
                    return [0, 2]
        elif action == 0:
            self.action_point = 0
            return [1, 2] 
        elif action == 1:
            self.action_point = 1
            return [0, 2]
        elif action == 3:
            self.action_point = 0
            return [1, 2]
        elif action == 4:
            self.action_point = 1
            return [0, 2]


    # 0 buy -> 1 sell / 2 hold
    # 1 sell -> 0 buy / 2 hold
    # 2 hold -> 0 buy / 1 sell
    # 3 con_buy
    # 4 con_sell

    # action point
    # action_point == 1 -> sell ~
    # action_point == 0 -> buy ~