import pandas as pd
from keras.layers import LSTM,Dense,Input
from keras.models import Model
from logging.handlers import RotatingFileHandler
import os
import time
import json
import logging
import sys
from sklearn.model_selection import train_test_split
import numpy as np


class Data:
    def __init__(self,config):
        self.config = config
        self.data,self.data_column_name = self.read_data()
       

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)
        print("train_num:",self.train_num)


        #需要进一步理解这段代码
        self.mean = np.mean(self.data,axis=0)
        self.std = np.std(self.data,axis=0) #每个特征的平方差
        self.norm_data = (self.data  - self.mean)/self.std  #然后对每个特征做归一化，这个归一化到比较小的数据值


        self.start_num_in_test = 0

    def read_data(self):
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path,nrows=self.config.debug_num,usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data_path,usecols=self.config.feature_columns)
        return init_data.values,init_data.columns.tolist()
    
    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day:self.config.predict_day + self.train_num,self.config.label_in_feature_index]
        print("label_data:")
        print(label_data)

        if not self.config.do_continue_train:
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            print("train_x list:",train_x)
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num - self.config.time_step)]
            print("train_y list:",train_y)
        else:
            #是一种比较奇怪的样本生成方式
            train_x = [feature_data[start_index + i*self.config.time_step:start_index + (i+1)*self.config.time_step]
            for start_index in range(self.config.time_step)
            for i in range((self.train_num - start_index)// self.config.time_step)]
            print("train_x:",type(train_x))

            train_y = [label_data[start_index + i*self.config.time_step:start_index + (i+1)*self.config.time_step]
            for start_index in range(self.config.time_step)
            for i in range((self.train_num - start_index) // self.config.time_step)]
            print("train_y:",type(train_y))
        train_x,train_y = np.array(train_x),np.array(train_y)
        print("train_x",train_x.size)
        print("train_y",train_y.size)

        train_x,valid_x,train_y,valid_y = train_test_split(train_x,train_y,test_size=self.config.test_data_rate,
                                                    random_state=self.config.random_seed,
                                                    shuffle=self.config.shuffle_train_data)

        print("train_x size:",train_x.shape)
        print("valid_x size:",valid_x.shape)
        print("train_y size:",train_y.shape)
        print("valid_y size:",valid_y.shape)
        return train_x,valid_x,train_y,valid_y

    
    def get_test_data(self,return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0],self.config.time_step) #防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval 

        test_x = [feature_data[self.start_num_in_test + i*sample_interval:self.start_num_in_test+(i+1)*sample_interval] for i in range(time_step_size)]
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:,self.config.label_in_feature_index]
            return np.array(test_x),label_data
        return np.array(test_x)


class Config:

    #load argument from json file
    file = open("config.json","r")
    config_obj = json.load(file)

    #feature_selected 
    feature_columns = config_obj["feature_selected"]["feature_column"]
    print("feature_columns:",feature_columns)
    label_columns = config_obj["feature_selected"]["label_column"]
    print("label_columns:",label_columns)
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始
    print(label_in_feature_index)
    


    #predict_day
    predict_day = config_obj["predict_day"]

    #network_arggument
    input_size = config_obj["network_argument"]["input_size"]
    output_size = config_obj["network_argument"]["output_size"]
    hidden_size = config_obj["network_argument"]["hidden_size"]
    lstm_layers = config_obj["network_argument"]["lstm_layers"]
    dropout_rate = config_obj["network_argument"]["dropout_rate"]
    time_step = config_obj["network_argument"]["time_step"]

    #train argument
    do_train = config_obj["train_argument"]["do_train"]
    do_predict = config_obj["train_argument"]["do_predict"]
    add_train = config_obj["train_argument"]["add_train"]
    shuffle_train_data = config_obj["train_argument"]["shuffle_train_data"]
    use_cuda = config_obj["train_argument"]["use_cuda"]
    train_data_rate = config_obj["train_argument"]["train_data_rate"]
    test_data_rate = config_obj["train_argument"]["test_data_rate"]
    batch_size = config_obj["train_argument"]["batch_size"]
    learning_rate = config_obj["train_argument"]["learning_rate"]
    epoch = config_obj["train_argument"]["epoch"]
    patience = config_obj["train_argument"]["patience"]
    random_seed = config_obj["train_argument"]["random_seed"]
    do_continue_train = config_obj["train_argument"]["do_continue_train"]
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    continue_flag = config_obj["train_argument"]["continue_flag"]

    #train mode
    debug_mode = config_obj["train_mode"]["debug_mode"]
    debug_num = config_obj["train_mode"]["debug_num"]
    user_frame =config_obj["train_mode"]["user_frame"]


    #path argument
    train_data_path = config_obj["path_argument"]["train_data_path"]
    model_save_path = config_obj["path_argument"]["model_save_path"]
    figure_save_path = config_obj["path_argument"]["figure_save_path"]
    log_save_path = config_obj["path_argument"]["log_save_path"]
    do_log_print_to_screen = config_obj["path_argument"]["do_log_print_to_screen"]
    do_log_save_to_file = config_obj["path_argument"]["do_log_save_to_file"]
    do_train_visualized = config_obj["path_argument"]["do_train_visualized"]

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())
        log_save_path = log_save_path + cur_time + "_" + user_frame + "/"
        os.makedirs(log_save_path)


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    #StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',fmt='[%(asctime)s] %(messge)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    #FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log",maxBytes=1024000,backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        #record config info to log file
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config,key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)
    return logger




def main(config):
    logger = load_logger(config)    
    try:
        np.random.seed(config.random_seed) #设置随机种子，保证可复现
        data_gainer = Data(config)
        if config.do_train:
            train_X,valid_X,train_Y,valid_Y = data_gainer.get_train_and_valid_data()
        
        if config.do_predict:
            test_X,test_Y = data_gainer.get_test_data(return_label_data=True)

    except Exception:
        logger.error("Run Error",exc_info=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #设置配置文件参数
    parser.add_argument("-t","--do_train",default=False,type=bool,help="whether to train")
    parser.add_argument("-p","--do_predict",default=True,type=bool,help="whether to predict")
    parser.add_argument("-b","--batch_size",default=64,type=int,help="batch size")
    parser.add_argument("-e","--epoch",default=20,type=int,help="epochs num")

    args = vars(parser.parse_args())
    do_train = args["do_train"]
    do_predict = args["do_predict"]
    batch_size = args["batch_size"]
    epoch = args["epoch"]

    #init the config of lstm
    con = Config()

    con.do_train = do_train
    con.do_predict = do_predict
    con.batch_size = batch_size
    con.epoch = epoch

    main(con)





