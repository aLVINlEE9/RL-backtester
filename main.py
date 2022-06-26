from datetime import date, datetime, timedelta
from data_preprocessor import Data_Generater
from environmnet import Environment
from learn import Learn
from db_parser import *

if __name__ == "__main__":
    
    train_value = [10, date(2021, 11, 10), 100] # 상위 10종목 date로부터 최근 10일간 code
    # epoch[0], state_size[1], action_size[2], trade money[3], discounting_factor[4], learning_rate[5], epsilon[6], 
    # epsilon_decay[7], epsilon_min[8], batch_size[9], train_start[10], max_memory[11]
    hyper_parameter = [100, 19, 5, 1000000, 0.99, 0.001, 1.0, \
                        0.999, 0.01, 500, 10000, 20000]

    # datagn = Data_Generater(train_value)
    # env = Environment(train_value)
    learn = Learn(train_value, hyper_parameter)

    # train_code_df = datagn.get_top_value_code() # 그 목록
    # train_code_np = datagn.get_top_value_db(train_code_df) # 해당 조건의 minute data를 전처리해서 각각 csv로 변환

    # 학습기 가동
    learn.run()
    

    #test_code = prep.get_top_value_code(10, date(2021, 11, 12), 2) # 상위 10종목 date로부터 최근 2일간(train * 0.2)
