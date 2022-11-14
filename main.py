import torch
import os
from fedmeta import asyfedmeta_reptile_gru,fedmeta_reptile_gru,fed_rnn,fed_gru,fed_lstm,asyfedmeta_reptile_lstm,asyfedmeta_reptile_rnn,fedmeta_reptile_lstm,fedmeta_reptile_rnn

def main(model_name,city,lr,time_len,num):
    epoch = 1003
    device_num = 3
    device = torch.device('cuda:{}'.format(device_num))
    folder = r"./model/model_{}".format(num)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = r"./model/model_{}/{}".format(num,city)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = os.path.join(folder,model_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = os.path.join(folder,"time_len_{}".format(time_len))
    if not os.path.exists(folder):
        os.mkdir(folder)
    if model_name == 'asyfedmeta_reptile_gru':
        meta_net = asyfedmeta_reptile_gru(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'asyfedmeta_reptile_lstm':
        meta_net = asyfedmeta_reptile_lstm(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'asyfedmeta_reptile_rnn':
        meta_net = asyfedmeta_reptile_rnn(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'fedmeta_reptile_gru':
        meta_net = fedmeta_reptile_gru(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'fedmeta_reptile_lstm':
        meta_net = fedmeta_reptile_lstm(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'fedmeta_reptile_rnn':
        meta_net = fedmeta_reptile_rnn(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'fed_gru':
        meta_net = fed_gru(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'fed_lstm':
        meta_net = fed_lstm(city,device,inner_lr=lr,time_len = time_len)
    if model_name == 'fed_rnn':
        meta_net = fed_rnn(city,device,inner_lr=lr,time_len = time_len)
    for i in range(1,1+epoch):
        print("{} round training.".format(i))
        if i == 0:
            torch.save({'model': meta_net.state_dict()},os.path.join(folder,"model_epoch_{}.pth".format(0)))
        meta_net.meta_training(epoch)
        if i%5 == 0:
            meta_net.Testing(i,num) 
            torch.save({'model': meta_net.state_dict()},os.path.join(folder,"model_epoch_{}.pth".format(i)))


if __name__ == '__main__':
    city = "gz" 
    lr = 0.001
    time_len = 6
    model_name = 'asyfedmeta_reptile_gru'
    
    for i in range(1,2):
        main(model_name,city,lr,time_len,i)