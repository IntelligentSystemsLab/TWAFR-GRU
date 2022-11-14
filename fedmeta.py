import torch
import numpy as np
from net import GRUNet,RNNNet,LSTMNet
from torch.autograd import Variable
import torch.nn as nn
import os
from client import Client


class asyfedmeta_reptile_gru(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,time_len = 3):
    super(asyfedmeta_reptile_gru, self).__init__()
    self.device = device
    self.city = city
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"

    for index,path in enumerate(train_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,round):
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_reptile_train()
        self.clients[j].epoch = round
      else:
        continue
    
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size

    weight = []
    for id,j in enumerate(id_train):
      weight.append(self.clients[j].size / size_all * np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()
    
    a = 0
    for id,j in enumerate(id_train):
      b = 0
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)

        w.data.add_(w_t.data*weight[id])


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/asyfedmeta_reptile_gru_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/asyfedmeta_reptile_gru_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/asyfedmeta_reptile_gru_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/asyfedmeta_reptile_gru_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/asyfedmeta_reptile_gru_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2) 


class asyfedmeta_reptile_lstm(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,time_len = 3):
    super(asyfedmeta_reptile_lstm, self).__init__()
    self.device = device
    self.city = city
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"

    for index,path in enumerate(train_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,round):
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_reptile_train()
        self.clients[j].epoch = round
      else:
        continue
    
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size

    weight = []
    for id,j in enumerate(id_train):
      weight.append(self.clients[j].size / size_all * np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()
    
    a = 0
    for id,j in enumerate(id_train):
      b = 0
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)

        w.data.add_(w_t.data*weight[id])


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/asyfedmeta_reptile_lstm_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/asyfedmeta_reptile_lstm_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/asyfedmeta_reptile_lstm_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/asyfedmeta_reptile_lstm_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/asyfedmeta_reptile_lstm_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2) 

class asyfedmeta_reptile_rnn(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,time_len = 3):
    super(asyfedmeta_reptile_rnn, self).__init__()
    self.device = device
    self.city = city
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"

    for index,path in enumerate(train_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,round):
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_reptile_train()
        self.clients[j].epoch = round
      else:
        continue
    
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size

    weight = []
    for id,j in enumerate(id_train):
      weight.append(self.clients[j].size / size_all * np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()
    
    a = 0
    for id,j in enumerate(id_train):
      b = 0
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)

        w.data.add_(w_t.data*weight[id])


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/asyfedmeta_reptile_rnn_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/asyfedmeta_reptile_rnn_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/asyfedmeta_reptile_rnn_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/asyfedmeta_reptile_rnn_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/asyfedmeta_reptile_rnn_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2) 

class asyfedmeta_fomaml_gru(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,time_len = 3):
    super(asyfedmeta_fomaml_gru, self).__init__()
    self.city = city
    self.device = device
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"

    for index,path in enumerate(train_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,round):
    
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_fomaml_train()
        self.clients[j].epoch = round
      else:
        continue
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size
    
    weight = []
    for id,j in enumerate(id_train):
      weight.append(self.clients[j].size / size_all * np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()

    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        
        w.data.add_(w_t.data*weight[id])



  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/asyfedmeta_fomaml_gru_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/asyfedmeta_fomaml_gru_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/asyfedmeta_fomaml_gru_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/asyfedmeta_fomaml_gru_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/asyfedmeta_fomaml_gru_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class asyfedmeta_fomaml_lstm(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,time_len = 3):
    super(asyfedmeta_fomaml_lstm, self).__init__()
    self.city = city
    self.device = device
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"

    for index,path in enumerate(train_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,round):
    
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_fomaml_train()
        self.clients[j].epoch = round
      else:
        continue
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size
    
    weight = []
    for id,j in enumerate(id_train):
      weight.append(self.clients[j].size / size_all * np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()

    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        
        w.data.add_(w_t.data*weight[id])



  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/asyfedmeta_fomaml_lstm_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/asyfedmeta_fomaml_lstm_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/asyfedmeta_fomaml_lstm_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/asyfedmeta_fomaml_lstm_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/asyfedmeta_fomaml_lstm_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class asyfedmeta_fomaml_rnn(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,time_len = 3):
    super(asyfedmeta_fomaml_rnn, self).__init__()
    self.city = city
    self.device = device
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"

    for index,path in enumerate(train_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,round):
    
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_fomaml_train()
        self.clients[j].epoch = round
      else:
        continue
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size
    
    weight = []
    for id,j in enumerate(id_train):
      weight.append(self.clients[j].size / size_all * np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()

    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        
        w.data.add_(w_t.data*weight[id])



  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/asyfedmeta_fomaml_rnn_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/asyfedmeta_fomaml_rnn_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/asyfedmeta_fomaml_rnn_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/asyfedmeta_fomaml_rnn_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/asyfedmeta_fomaml_rnn_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fedmeta_reptile_gru(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,batch_size = 100,time_len = 3):
    super(fedmeta_reptile_gru, self).__init__()
    self.city = city
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.net = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"

    for index,path in enumerate(train_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_meta_reptile_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fedmeta_reptile_gru_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fedmeta_reptile_gru_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fedmeta_reptile_gru_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fedmeta_reptile_gru_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fedmeta_reptile_gru_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fedmeta_reptile_rnn(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,batch_size = 100,time_len = 3):
    super(fedmeta_reptile_rnn, self).__init__()
    self.city = city
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.net = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"

    for index,path in enumerate(train_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_meta_reptile_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fedmeta_reptile_rnn_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fedmeta_reptile_rnn_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fedmeta_reptile_rnn_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fedmeta_reptile_rnn_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fedmeta_reptile_rnn_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fedmeta_reptile_lstm(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=1,inner_lr=0.001,batch_size = 100,time_len = 3):
    super(fedmeta_reptile_lstm, self).__init__()
    self.city = city
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.net = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"

    for index,path in enumerate(train_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_meta_reptile_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fedmeta_reptile_lstm_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fedmeta_reptile_lstm_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fedmeta_reptile_lstm_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fedmeta_reptile_lstm_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fedmeta_reptile_lstm_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fedmeta_fomaml_gru(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=0.001,inner_lr=0.001,batch_size = 100,time_len = 3):
    super(fedmeta_fomaml_gru, self).__init__()
    self.city = city
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.net = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}_new_save/train".format(city)
    test_path = r"./dataset/{}_new_save/test".format(city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"

    for index,path in enumerate(train_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_meta_fomaml_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fedmeta_fomaml_gru_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fedmeta_fomaml_gru_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fedmeta_fomaml_gru_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fedmeta_fomaml_gru_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fedmeta_fomaml_gru_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)
  
class fedmeta_fomaml_lstm(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=0.001,inner_lr=0.001,batch_size = 100,time_len = 3):
    super(fedmeta_fomaml_lstm, self).__init__()
    self.city = city
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.net = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}_new_save/train".format(city)
    test_path = r"./dataset/{}_new_save/test".format(city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"

    for index,path in enumerate(train_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_meta_fomaml_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fedmeta_fomaml_lstm_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fedmeta_fomaml_lstm_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fedmeta_fomaml_lstm_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fedmeta_fomaml_lstm_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fedmeta_fomaml_lstm_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fedmeta_fomaml_rnn(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=0.001,inner_lr=0.001,batch_size = 100,time_len = 3):
    super(fedmeta_fomaml_rnn, self).__init__()
    self.city = city
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.net = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}_new_save/train".format(city)
    test_path = r"./dataset/{}_new_save/test".format(city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"

    for index,path in enumerate(train_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_meta_fomaml_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))


  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fedmeta_fomaml_rnn_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fedmeta_fomaml_rnn_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fedmeta_fomaml_rnn_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fedmeta_fomaml_rnn_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fedmeta_fomaml_rnn_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fed_rnn(nn.Module):
  def __init__(self, city,device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=0.001,inner_lr=0.001,time_len = 3):
    super(fed_rnn, self).__init__()
    self.city = city
    self.device = device
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fed_train"
    self.mode_2 = "fed_test"

    for index,path in enumerate(train_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = RNNNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_fed_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))



  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fed_rnn_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fed_rnn_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fed_rnn_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fed_rnn_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fed_rnn_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fed_lstm(nn.Module):
  def __init__(self,city, device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=0.001,inner_lr=0.001,time_len = 3):
    super(fed_lstm, self).__init__()
    self.city = city
    self.device = device
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fed_train"
    self.mode_2 = "fed_test"

    for index,path in enumerate(train_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = LSTMNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_fed_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))

  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fed_lstm_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fed_lstm_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fed_lstm_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fed_lstm_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fed_lstm_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)

class fed_gru(nn.Module):
  def __init__(self,city, device,local_metatrain_epoch=1, local_test_epoch=10,outer_lr=0.001,inner_lr=0.001,time_len = 3):
    super(fed_gru, self).__init__()
    self.city = city
    self.device = device
    self.inner_lr = inner_lr
    self.time_len = time_len
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
    self.loss_function = torch.nn.MSELoss()
    train_path = r"./dataset/{}/train".format(self.city)
    test_path = r"./dataset/{}/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fed_train"
    self.mode_2 = "fed_test"

    for index,path in enumerate(train_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = GRUNet(input_size=1, hidden_size=1,
                          seq_len=6, output_size=1, num_layers=1).to(self.device)
      self.test_clients.append(Client(model,index,path,local_metatrain_epoch,local_test_epoch,time_len,inner_lr,outer_lr,self.device,self.mode_2))

  def forward(self):
    pass

  def meta_training(self,epoch):
    id_train = list(range(len(self.clients)))
    for id,j in enumerate(id_train):
      self.clients[j].refresh(self.net)
      self.clients[j].local_fed_train()
    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data)
    for w in self.net.parameters():
      w.data.div_(len(id_train))



  def Testing(self,round,num):
    id_test = list(range(len(self.test_clients)))
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list = self.test_clients[id].test()
      if a == 0:
        final_test_loss = test_loss_list.copy()
        final_test_mae = test_mae_list.copy()
        final_test_rmse = test_rmse_list.copy()
        final_test_rae = test_rae_list.copy()
        final_test_r2 = test_r2_list.copy()
      else:
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_test_mae = np.concatenate((final_test_mae,test_mae_list),axis = 0)
        final_test_rmse = np.concatenate((final_test_rmse,test_rmse_list),axis = 0)
        final_test_rae = np.concatenate((final_test_rae,test_rae_list),axis = 0)
        final_test_r2 = np.concatenate((final_test_r2,test_r2_list),axis = 0)

    folder_new_new = r"./result/result_new_{}".format(num)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}".format(num,self.city)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/loss".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/mae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rmse".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/rae".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)
    folder_new_new = r"./result/result_new_{}/{}/{}/r2".format(num,self.city,self.time_len)
    if not os.path.exists(folder_new_new):
      os.mkdir(folder_new_new)  
    loss_test_file_path = r"./result/result_new_{}/{}/{}/loss/fed_gru_test_loss".format(num,self.city,self.time_len)
    mae_test_file_path = r"./result/result_new_{}/{}/{}/mae/fed_gru_test_mae".format(num,self.city,self.time_len)
    rmse_test_file_path = r"./result/result_new_{}/{}/{}/rmse/fed_gru_test_rmse".format(num,self.city,self.time_len)
    rae_test_file_path = r"./result/result_new_{}/{}/{}/rae/fed_gru_test_rae".format(num,self.city,self.time_len)
    r2_test_file_path = r"./result/result_new_{}/{}/{}/r2/fed_gru_test_r2".format(num,self.city,self.time_len)
    
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(mae_test_file_path):
      os.mkdir(mae_test_file_path)
    if not os.path.exists(rmse_test_file_path):
      os.mkdir(rmse_test_file_path)
    if not os.path.exists(rae_test_file_path):
      os.mkdir(rae_test_file_path)
    if not os.path.exists(r2_test_file_path):
      os.mkdir(r2_test_file_path)
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    mae_test_path = os.path.join(mae_test_file_path,"{}.npy".format(round))
    rmse_test_path = os.path.join(rmse_test_file_path,"{}.npy".format(round))
    rae_test_path = os.path.join(rae_test_file_path,"{}.npy".format(round))
    r2_test_path = os.path.join(r2_test_file_path,"{}.npy".format(round))
    np.save(loss_test_path,final_test_loss) 
    np.save(mae_test_path,final_test_mae) 
    np.save(rmse_test_path,final_test_rmse) 
    np.save(rae_test_path,final_test_rae)
    np.save(r2_test_path,final_test_r2)
