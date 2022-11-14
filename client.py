import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from data_reader import MyData
from copy import deepcopy,copy
from torch.autograd import Variable 
from torch.utils.data import DataLoader
import pandas as pd
import time

class Client(nn.Module):
    def __init__(self,model,id,path,update_step,update_step_test,time_len,base_lr,meta_lr,device,mode):
        super(Client, self).__init__()
        self.id = id
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.net = deepcopy(model)
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        data = pd.read_excel(path, engine='openpyxl')
        data["percent"] = data["busy"]/data["total"]
        data = data['percent'].values.astype('float64')
        data = torch.as_tensor(data, device=device).float().to(device)
        self.mode = mode
        self.time = 0
        self.epoch = 0
        if self.mode == "fed_train":
            support_size = int(len(data)*1.0)
            support_target = data[:support_size]
            support_set = MyData(support_target, seq_length=6,time_len = time_len,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=len(support_set), shuffle=False)
            
        elif self.mode == "reptile_train":
            support_size = int(len(data)*1.0)
            support_target = data[:support_size]
            support_set = MyData(support_target, seq_length=6,time_len = time_len,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=int(0.2*len(support_set)), shuffle=False)
        else:
            support_size = int(len(data)*0.8)
            support_target = data[:support_size]
            support_set = MyData(support_target, seq_length=6,time_len = time_len,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=len(support_set), shuffle=False)

        query_size = int(len(data)*1.0)


        if self.mode == "fed_train":
            pass
        elif self.mode == "reptile_train":
            pass
        else:
            query_target = data[support_size:query_size]
            query_set = MyData(query_target, seq_length=6,time_len = time_len,device = device)
            self.query_loader = DataLoader(
                query_set, batch_size=len(query_set), shuffle=False)
        self.optim = torch.optim.Adam(self.net.parameters(), lr = self.base_lr)
        self.outer_optim = torch.optim.SGD(self.net.parameters(), lr = self.meta_lr)
        self.size = query_size
        self.device = device
        self.loss_function = torch.nn.MSELoss().to(self.device)
        
    def forward(self):
        pass

    def local_meta_reptile_train(self):
        for _ in range(self.update_step):
            net_tem = deepcopy(self.net)
            meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            i = 1
            for support in self.support_loader:
                support_x, support_y = support
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                
                meta_optim_tem.zero_grad()
                output = net_tem(support_x,)
                output = torch.squeeze(output)
                loss = self.loss_function(output,support_y)
                loss.backward()
                meta_optim_tem.step()

                i += 1
                if i > 5:
                    break

            self.outer_optim.zero_grad()
            for w, w_t in zip(self.net.parameters(), net_tem.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(self.device)
                w.grad.data.add_(w.data - w_t.data)

            self.outer_optim.step()
    
    def local_meta_fomaml_train(self):
        for _ in range(self.update_step):
            for support,query in zip(self.support_loader,self.query_loader):
                support_x, support_y = support
                query_x, query_y = query
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                
                
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output,support_y) 
                
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                self.outer_optim.zero_grad()
                output = self.net(query_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output, query_y)
                
                loss.backward()
                self.outer_optim.step()
                self.outer_optim.zero_grad()
                break
    

    def local_asymeta_fomaml_train(self):
        base_link_rate = np.random.randint(1300, 4500, (2,))
        link_delay = np.random.randint(1, 10, (2,))
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.update_step):
            for support,query in zip(self.support_loader,self.query_loader):
                support_x, support_y = support
                query_x, query_y = query
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                
                
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output,support_y) 
                
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                self.outer_optim.zero_grad()
                output = self.net(query_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output, query_y)
                
                loss.backward()
                self.outer_optim.step()
                self.outer_optim.zero_grad()
                break
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission

        self.time = t_all
    
    def local_asymeta_reptile_train(self,):
        base_link_rate = np.random.randint(1300, 4500, (2,))

        link_delay = np.random.randint(1, 10, (2,))
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.update_step):
            net_tem = deepcopy(self.net)
            meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            i = 1
            for support in self.support_loader:
                support_x, support_y = support
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                
                meta_optim_tem.zero_grad()
                output = net_tem(support_x,)
                output = torch.squeeze(output)
                loss = self.loss_function(output,support_y)
                loss.backward()
                meta_optim_tem.step()

                i += 1
                if i > 5:
                    break

            self.outer_optim.zero_grad()
            for w, w_t in zip(self.net.parameters(), net_tem.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(self.device)
                w.grad.data.add_(w.data - w_t.data)

            self.outer_optim.step()
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission

        self.time = t_all

    def local_fed_train(self):
        for _ in range(self.update_step):
            for support in self.support_loader:
                support_x, support_y = support
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output, support_y)
                
                loss.backward()
                self.optim.step()
                break

        self.optim.zero_grad()

    def refresh(self,model):
        for w,w_t in zip(self.net.parameters(),model.parameters()):
            w.data.copy_(w_t.data)
        
    def test(self):
        test_loss_list = torch.zeros([self.update_step_test+1])
        test_mae_list = torch.zeros([self.update_step_test+1])
        test_rmse_list = torch.zeros([self.update_step_test+1])
        test_rae_list = torch.zeros([self.update_step_test+1])
        test_r2_list = torch.zeros([self.update_step_test+1])

        for query in self.query_loader:
            query_x, query_y = query
            if torch.cuda.is_available():
                query_x = query_x.cuda(self.device)
                query_y = query_y.cuda(self.device)
            output = self.net(query_x)
            output = torch.squeeze(output)
            loss = self.loss_function(output, query_y)
            mae = torch.abs(output - query_y)
            mae = torch.mean(mae)
            rmse = (output - query_y)*(output - query_y)
            rmse = torch.sqrt(torch.mean(rmse.data))
            rae_1 = torch.abs(output - query_y).sum()
            mean = output.mean()
            rae_2 = torch.abs(query_y - mean).sum()
            rae = rae_1/rae_2
            r_2_1 = (output - query_y)*(output - query_y)
            r_2_1 = r_2_1.sum()
            r_2_2 = (query_y - mean)*(query_y - mean)
            r_2_2 = r_2_2.sum()
            r_2 = 1-(r_2_1/r_2_2)
            
    
            test_loss_list[0] = loss.item()
            test_mae_list[0] = mae.item()
            test_rmse_list[0] = rmse.item()
            test_rae_list[0] = rae.item()
            test_r2_list[0] = r_2.item()

        print("{} Fine Tuning!".format(self.id))
        for epoch in range(1,self.update_step_test+1):
            for support in self.support_loader:
                self.optim.zero_grad()
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                output = self.net(support_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()


            for query in self.query_loader:
                query_x, query_y = query
                if torch.cuda.is_available():
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                output = self.net(query_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output, query_y)
                mae = torch.abs(output - query_y)
                mae = torch.mean(mae)
                rmse = (output - query_y)*(output - query_y)
                rmse = torch.sqrt(torch.mean(rmse.data))
                rae_1 = torch.abs(output - query_y).sum()
                mean = output.mean()
                rae_2 = torch.abs(query_y - mean).sum()
                rae = rae_1/rae_2
                r_2_1 = (output - query_y)*(output - query_y)
                r_2_1 = r_2_1.sum()
                r_2_2 = (query_y - mean)*(query_y - mean)
                r_2_2 = r_2_2.sum()
                r_2 = 1-(r_2_1/r_2_2)
                
        
                test_loss_list[epoch] = loss.item()
                test_mae_list[epoch] = mae.item()
                test_rmse_list[epoch] = rmse.item()
                test_rae_list[epoch] = rae.item()
                test_r2_list[epoch] = r_2.item()

        test_loss_list = test_loss_list.detach().numpy()
        test_mae_list = test_mae_list.detach().numpy()
        test_rmse_list = test_rmse_list.detach().numpy()
        test_rae_list = test_rae_list.detach().numpy()
        test_r2_list = test_r2_list.detach().numpy()
        
        return test_loss_list,test_mae_list,test_rmse_list,test_rae_list,test_r2_list

        
    

        


    
    
 