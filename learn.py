from collections import deque
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

class Learn:
    def __init__(self, train_value, hyper_parameter):
        self.env = Environment(train_value)
        self.agent = Agent(hyper_parameter, self.env)
        self.visual = Visualizer()
        self.train_value = train_value

        # 하이퍼 파라미터 리스트 분해
        # epoch[0], state_size[1], action_size[2], trade money[3], discounting_factor[4], learning_rate[5], epsilon[6], 
        # epsilon_decay[7], epsilon_min[8], batch_size[9], train_start[10], max_memory[11]
        self.epoch = hyper_parameter[0]
        self.state_size = hyper_parameter[1]
        self.action_size = hyper_parameter[2]
        self.trade_money = hyper_parameter[3]
        self.discounting_factor = hyper_parameter[4]
        self.learning_rate = hyper_parameter[5]
        self.epsilon = hyper_parameter[6]
        self.epsilon_decay = hyper_parameter[7]
        self.epsilon_min = hyper_parameter[8]
        self.batch_size = hyper_parameter[9]
        self.train_start = hyper_parameter[10]
        self.max_memory = hyper_parameter[11]


        # model 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 리플레이 메모리
        self.memory = deque(maxlen=self.max_memory)

        # 타깃 모델 초기화
        self.update_target_model()

        # replay memory 
        # (?) replay memory 에 q 함수 값 들어감
        self.memory_state = []
        self.memory_reward = []
        self.memory_action = []
        self.memory_next_state = []



    def build_model(self):
        model = Sequential()
        # np.array [list([1], hold reward) list([2]) list([3]) list([4]) list([5]) list([6]) ]
        model.add(Dense(100, input_shape=(6, 19), activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(100, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(100, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, 6, self.state_size))
        next_states = np.zeros((self.batch_size, 6, self.state_size))
        actions, rewards = [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if actions[i] == 1:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discounting_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def append_sample(self, action_p, sample_list):
        if action_p == 1: # <buy - hold - sell> sell ~
            # agent 정보를 reset한다.
            self.agent.reset_agent()
            # reward 수정
            #print(sample_list)
            sample_list = np.array(sample_list, dtype="object") # dtype="object"
            #print(sample_list)
            states = sample_list[:,0]
            actions = sample_list[:,1]
            rewards = sample_list[:,2]
            next_states = sample_list[:,3]
            length_list = len(sample_list)
            #print(states, actions, rewards, next_states)

            fin_reward = rewards[length_list - 1]
            for i in range(length_list):
                cur_reward = rewards[i]
                if cur_reward > 0:
                    if cur_reward > fin_reward:
                        ret_reward = fin_reward - cur_reward
                    elif cur_reward <= fin_reward:
                        ret_reward = cur_reward
                elif cur_reward <= 0:
                    ret_reward = cur_reward
                # 수정된 sample memory에 append
                self.memory.append((states[i], actions[i], ret_reward, next_states[i]))
            return True

        else:
            return False


    # 0 buy -> 1 sell / 2 hold
    # 1 sell -> 0 buy / 2 hold
    # 2 hold -> 0 buy / 1 sell / 2 hold

    # action_point == 1 -> sell ~
    # action_point == 0 -> buy ~

    def run(self):
        #code 정보 불러오기
        try:
            code_df = pd.read_csv(f"{SYSTEM_PATH}/traindata_files/"
                                f"top[{self.train_value[0]}]recent[{self.train_value[2]}]end_d[{self.train_value[1]}]/"
                                f"code[top{self.train_value[0]}_recent{self.train_value[2]}days_{self.train_value[1]}].csv")
            code_df['code'] = code_df['code'].apply(lambda x : str(x).zfill(6))
            code_np = code_df[['code', 'date']].to_numpy()
        

            profit_sum = 0
            profit_list, stock_n_list = [], []

            # 각 종목별로 observation 하기 (epoch)
            for stock_n in range(len(code_np)):

                try:
                    # data 로드
                    minute_db, length = self.env.load_data(code_np, stock_n)

                    # agent 정보 reset
                    self.agent.reset_agent()
                    # 첫 state 설정
                    state = self.agent.get_state(self.env.observe(minute_db, 0), -1, 0)
                    
                    # sample들을 임시 저장할 리스트 생성
                    sample_list = []

                    self.visual.data_set(minute_db)
                    buy_hold_cnt = 0
                    sell_hold_cnt = 0
                    action_point = -1
                    profit = 0
                    # 한 종목 별로
                    for n in range(length):
                        # 해당 시간(분)에 대한 정보
                        observation = self.env.observe(minute_db, n) # [idx, code, date, time, price, data]
                        #print(f"observe{observation[1]}--------------------------={n}")

                        if action_point == 0: # buy~
                            buy_hold_cnt += 1
                            sell_hold_cnt = 0
                        elif action_point == 1 or action_point == -1: # sell~ 
                            buy_hold_cnt = 0
                            sell_hold_cnt += 1   
                        # action (epsilon-greedy policy로 불러옴)
                        # action_point(지금 action이 after buy[1]인지 after sell[1]인지 확인)
                        action, action_point = self.agent.get_action(state, buy_hold_cnt, sell_hold_cnt, self.model)
                        #print(f"get_action{action} {action_point}--------------------------={n}")


                        # agent가 해당 행동을 하였을때 대한 reward 반환
                        immediate_reward, profit, action = self.agent.act(action, buy_hold_cnt, sell_hold_cnt, observation) # im_re : 즉시 보상, dly_re : 지연 보상
                        #print(f"act{immediate_reward} {profit}--------------------------={n}")
                        #print(action)
                        #print("")

                        self.visual.throw_action(action, profit)

     

                        # next state를 가져옴
                        if length == n + 1:
                            if action == 1:
                                next_state = self.agent.get_state(self.env.observe(minute_db, n), action_point, immediate_reward)
                        else:
                            next_state = self.agent.get_state(self.env.observe(minute_db, n + 1), action_point, immediate_reward)
                        #print(f"get_state{state[5][18]} {next_state[5][18]}--------------------------={n}")
                        
                        # replay memory에 저장하는 단계
                        # if -> buy - hold - sell 임시 sample_list 초기화
                        # else -> 임시 sample_list append
                        if action_point == 0:
                            sample_list.append([state, action, immediate_reward, next_state])
                        #print(f"append{len(sample_list)}--------------------------={n}")

                        if self.append_sample(action, sample_list):
                            profit_sum += profit
                            sample_list = []
                        
                        # train_start만큼 채워지고 train_start buy - hold - sell 끝나면 학습
                        if len(self.memory) >= self.train_start and action == 1:
                            self.train_model()
                        
                        state = next_state
                        #print("stock:", stock_n, "n:", n, "reward:", immediate_reward)
                    
                    # 한 종목을 
                    self.update_target_model()

                    profit_list.append(profit_sum)
                    stock_n_list.append(stock_n)
                    print("stock:", stock_n, "  profit_sum:", profit_sum, "  memory length:",
                      len(self.memory), "  epsilon:", self.epsilon)

                    self.visual.plot_graph(stock_n, "train")

                except Exception as e:
                    print(f"Exception occured : {e} ({stock_n})")


                self.model.save_weights("./stock_dqn.h5")
                pylab.figure(len(code_np))
                pylab.plot(stock_n_list, profit_list, 'b')
                pylab.savefig("./stock_dqn.png")

        except Exception as e:
            print(f"Exception occured: {e}")