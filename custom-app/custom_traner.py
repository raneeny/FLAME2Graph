# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:58:13 2022

@author: Raneen_new
"""
import os.path
import dill
#import torch
#from pt_constants import PTConstants
#from simple_network import SimpleNetwork
#from torch import nn
#from torch.optim import SGD
#from torch.utils.data.dataloader import DataLoader
#from torch.utils.tensorboard import SummaryWriter
#from torchvision.datasets import CIFAR10
#from torchvision.transforms import Compose, Normalize, ToTensor
################ our imports- Raneen Code############################
from Data_Preprocessing import ReadData
from ConvNet_Model import ConvNet
import numpy as np
import tensorflow.keras as keras
from High_Activated_Filters import HighlyActivated
from Clustering import Clustering
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import random
from Embading import Graph_embading
import time
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from numpy import mean
from numpy import std
import networkx as nx
np.random.seed(0)
import pickle
import shutil
###################################flare imports####################



from nvflare.apis.dxo import DXO, DataKind, from_shareable, MetaKey
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants





def _timeseries_embedding(embedding_graph,node_names,timesereis_MHAP,number_seg):
    feature_list = []
    embed_vector = embedding_graph.wv[node_names]
    for i,data in enumerate(timesereis_MHAP):
        #compare the name with word_list and take its embedding
        #loop through segmant
        segmant = [[] for i in range(number_seg)]
        #print(len(data))
        for m,seg in enumerate(data):
            temp = [0 for i in range(len(embed_vector[0]))]
            #each seg has mhaps
            for k,mhap in enumerate(seg):
                for j,node in enumerate(node_names):
                    if(mhap == node):
                        temp += embed_vector[j]
                        break
            segmant[m].append(list(temp))
        feature_list.append(segmant)
    return feature_list



def _readData(data_name,dir_name):
     dir_path = dir_name + data_name+'/'
     dataset_path = dir_path + data_name +'.mat'
     # print('*******************************************')
     # print(os.path)
     # print(os.getcwd())

     ##read data and process it
     prepare_data = ReadData()
     prepare_data.data_preparation(dataset_path, dir_path)
     datasets_dict = prepare_data.read_dataset(dir_path,data_name)
     x_train = datasets_dict[data_name][0]
     y_train = datasets_dict[data_name][1]
     x_test = datasets_dict[data_name][2]
     y_test = datasets_dict[data_name][3]
     x_train, x_test = prepare_data.z_norm(x_train, x_test)
     nb_classes = prepare_data.num_classes(y_train,y_test)
     y_train, y_test, y_true = prepare_data.on_hot_encode(y_train,y_test)
     x_train, x_test, input_shape = prepare_data.reshape_x(x_train,x_test)
     x_training = x_train
     y_training = y_train
     
     x_new1 = np.concatenate((x_train, x_test), axis=0)
     y_new1 = np.concatenate((y_train, y_test), axis=0)
     x_training, x_validation, y_training, y_validation = train_test_split(x_new1, y_new1, test_size=0.20,shuffle=True)
     x_validation,x_test,y_validation,y_test = train_test_split(x_validation, y_validation, test_size=0.50,shuffle=True)

     return x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes
 
def _downsample_to_proportion(rows, proportion=1):
        i = 0
        new_data = []
        new_data.append(rows[0])
        k = 0
        for i in (rows):
            if(k == proportion):
                new_data.append(i)
                k = 0
            k+=1
        return new_data 
def _normilization(data):
              i = 0
              datt = []
              maxi = max(data)
              mini = abs(min(data))
              while (i< len(data)):
                  
                  if(data[i] >=0):
                      val = data[i]/maxi
                  else:
                      val = data[i]/mini
               
                  datt.append(val)
                  i += 1
                  
              return datt   
def _fitted_cluster(data,cluster):
        print(len(cluster))
        print(len(data))
        data = _normilization(data)
        cluster[0] = _normilization(cluster[0])
        #print(data)
        #print(cluster[0])
        data = np.nan_to_num(data)
        cluster[0] = np.nan_to_num(cluster[0])
        mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = _normilization(i)
            data = np.nan_to_num(data)
            clu_nor = np.nan_to_num(clu_nor)
            dist = distance.euclidean(data,clu_nor)
            #print(dist)
            if(dist < mini):
                cluster_id = count
                mini = dist
            count+=1   
        return cluster_id
    
    
    
def _similarity_array(x,y):
          sim = [0]*len(x)
          cp_x = np.copy(x)
          for j in range(len(y)):
              index_sim = _fitted_cluster(y[j],cp_x)
              for i in range(len(x)):
                  if(np.array_equal(cp_x[index_sim],x[i])):    
                      sim[j] = i
                      break
              cp_x = np.delete(cp_x, index_sim, axis=0)

          return sim
          
def _similirity_node_name(centriods):
    i = 1
    clients_sim = []
    #for first client make it default
    cl_simi = []
    #shuffle each time centriod index
    ind_cent=random.randint(0, len(centriods)-1)
    new_cent = []
    new_cent.append(centriods[ind_cent])
    for j in range(len(centriods)):
        if(j != ind_cent):
            new_cent.append(centriods[j])
    centriods = new_cent
    for k in range (len(centriods[0])):
        simi_client = []
        for l in range(len(centriods[0][k])):
            simi_client.append(l)
        cl_simi.append(simi_client)
    clients_sim.append(cl_simi)
    #loop through clients centriod list
    while(i < len(centriods)):
        #loop through layers
        simi_client = []
        for k in range (len(centriods[0])):
            layers = centriods[0][k]
            print(layers)
            simi_client.append(_similarity_array(layers,centriods[i][k]))
        clients_sim.append(simi_client)
        i +=1
    #print(clients_sim)
    return clients_sim,ind_cent

def _relabiling_client_graph(graph,maping):  
        mapping_graph = {}
        node_name = list(graph.nodes)
        for i in node_name:
            new_name = maping[int(i[5])][int(i[7])]
            new_name = 'layer%s %s' %(i[5],new_name)
            mapping_graph[i] = new_name
        G = nx.relabel_nodes(graph, mapping_graph)
        return G

def _server_graph_aggregation(graphs,centriods,):
         #find the node similarity between all the clients local graphs
         #find the node similarity between all the clients local graphs
        new_cent_index,ind_cent = _similirity_node_name(centriods)
        #relabiling the graphs based on the new similarity
        new_graph = []
        for i in range(len(graphs)):
            new_graph.append(_relabiling_client_graph(graphs[i],new_cent_index[i]))
        #comose the new clients graph into a new one
        Graph1 = new_graph[ind_cent]
        i = 1
        global_graph=nx.Graph()
        while i < len(new_graph):
            global_graph = nx.compose(Graph1,new_graph[i])
            i+=1
        return global_graph,centriods[ind_cent] 
def _clusering_array(period_active,comm_round):
     cluser_data_pre_list = []
     filter_lists = [[] for i in range(3)]
     for i in range(len(period_active)):
         for j in range(len(period_active[i])):
             for k in range(len(period_active[i][j])):
                 filter_lists[j].append(period_active[i][j][k])

     cluser_data_pre_list.append([x for x in filter_lists[0] if x])
     cluser_data_pre_list.append([x for x in filter_lists[1] if x])
     cluser_data_pre_list.append([x for x in filter_lists[2] if x])
     print(len(cluser_data_pre_list[0]))
     print(len(cluser_data_pre_list[1]))
     print(len(cluser_data_pre_list[2]))
     
     cluser_data_pre_list1 = []
     
     cluser_data_pre_list1.append(_downsample_to_proportion(cluser_data_pre_list[0], 1000))
     cluser_data_pre_list1.append(_downsample_to_proportion(cluser_data_pre_list[1], 1000))
     cluser_data_pre_list1.append(_downsample_to_proportion(cluser_data_pre_list[2], 1000))
     cluser_data_pre_list1 = np.array(cluser_data_pre_list1)

     clustering = Clustering(cluser_data_pre_list1)
     cluser_data_pre_list1 = clustering.scale_data(cluser_data_pre_list1)
     clustering = Clustering(cluser_data_pre_list1)
     c_num = [35,25,15]
     if(comm_round <= 5):
        c_num = [5,5,5]
     elif(5<comm_round <= 10):
        c_num = [10,8,8]
     elif(10<comm_round <= 18):
        c_num = [15,12,12]
     else:
        c_num = [35,25,17]
     cluster_central = clustering.cluster_sequence_data(c_num,[8,40,120],cluser_data_pre_list1)
     return cluster_central

    
def _trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes,epoch,weight,flag):
     ##train the model
     train_model = ConvNet()
     model = train_model.network_fcN(input_shape,nb_classes,weight,flag)
     train_model.trainNet(model,x_training,y_training,x_validation,y_validation,16,epoch)
     return model,train_model

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



class SimpleTrainer(Executor):
    
   def __init__(self, epochs_per_round=6,  train_task_name=AppConstants.TASK_TRAIN,
                validate_task_name=AppConstants.TASK_VALIDATION):
        super().__init__()
        
        self.logger.info("+++++++++++Iniliaastion of simpletranier")
        self._train_task_name = train_task_name
        self._validate_task_name = validate_task_name
        self.epochs_per_round = epochs_per_round
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.model = None
        self.numRound=0
        self._global_graph=[]
        self._centriod=[]

    # tell when setup is need to be called -- before the client recieve model
   def handle_event(self, event_type: str, fl_ctx: FLContext):
        print(event_type)
        if event_type == EventType.START_RUN:
            self.logger.info("+++++++++++ setup ")
            self.setup(fl_ctx)
            
            
        if event_type == EventType.ABOUT_TO_END_RUN:
             self.logger.info("+++++++++++ clean ")
             self.clean(fl_ctx)
                    
        # if event_type == EventType.After_SEND_TASK_RESULT:

        #      self.logger.info(f"+++++++++++ clean ")
        #      self.validate(fl_ctx)
        
            
   def clean(self,fl_ctx):
       dir='%s_model_' %(self.client_name)
       if  os.path.isdir(dir):
           shutil.rmtree(dir)
       else:
           self.logger.info("+++++++++++ no module folder found to be deleted ")
           
      
      
       
   
            
   def setup(self, fl_ctx: FLContext):

       #prepare data
       data_name = 'Wafer'
       dir_name = './data/'
       x_train, x_validation, x_test, y_train, y_validation,y_true,y_test,input_shape, nb_classes = _readData(data_name,dir_name)
       self.train_images = np.concatenate((x_train, x_validation), axis=0)
       self.train_labels = np.concatenate((y_train, y_validation), axis=0)
       self.test_images= x_test
       self.test_labels= y_test
       self.input_shape=input_shape
       self.nb_classes= nb_classes
    
      
      #data,label = merge_traning_testing_data(x_training, x_validation, x_test, y_training, y_validation, y_test)


        # simulate separate datasets for each client by dividing MNIST dataset in half
       self.client_name = fl_ctx.get_identity_name()
       
       chuckLength= len(self.train_images) // 4
       train_chunks=list(_chunks(self.train_images,chuckLength))
       trainLabel_chunks=list(_chunks(self.train_labels,chuckLength))
       
       chuckLength= len(self.test_images) 
       test_chunks= list(_chunks(self.test_images,chuckLength))
       testLabel_chunks=list(_chunks(self.test_labels,chuckLength))
       name_x = '%s_X_test.npy' %(fl_ctx.get_identity_name())
       name_y = '%s_Y_test.npy' %(fl_ctx.get_identity_name())
       
       
       if self.client_name == "site-1":
           self.train_images = train_chunks[0]
           self.train_labels = trainLabel_chunks[0]
           self.test_images = test_chunks[0]
           self.test_labels = testLabel_chunks[0]          
           np.save(name_x,self.test_images)
           np.save(name_y,self.test_labels)
           
           
           
           
           # self.train_images = self.train_images[: len(self.train_images) // 4]
           # self.train_labels = self.train_labels[: len(self.train_labels) // 2]
           # self.test_images = self.test_images[: len(self.test_images) // 2]
           # self.test_labels = self.test_labels[: len(self.test_labels) // 2]
       elif self.client_name == "site-2":
           self.train_images = train_chunks[1]
           self.train_labels = trainLabel_chunks[1]
           self.test_images = test_chunks[0]
           self.test_labels = testLabel_chunks[0]
           np.save(name_x,self.test_images)
           np.save(name_y,self.test_labels)
           
         
           
           # self.train_images = self.train_images[len(self.train_images) // 2 :]
           # self.train_labels = self.train_labels[len(self.train_labels) // 2 :]
           # self.test_images = self.test_images[len(self.test_images) // 2 :]
           # self.test_labels = self.test_labels[len(self.test_labels) // 2 :]
           
       elif self.client_name == "site-3":
            self.train_images = train_chunks[2]
            self.train_labels = trainLabel_chunks[2]
            self.test_images = test_chunks[0]
            self.test_labels = testLabel_chunks[0]
            np.save(name_x,self.test_images)
            np.save(name_y,self.test_labels)
            
           
       elif self.client_name == "site-4":
                 self.train_images = train_chunks[3]
                 self.train_labels = trainLabel_chunks[3]
                 self.test_images = test_chunks[0]
                 self.test_labels = testLabel_chunks[0]
                 np.save(name_x,self.test_images)
                 np.save(name_y,self.test_labels)
                 
       #model,train_model = trainModel(self.train_images, self.test_images, self.train_labels, self.test_labels,input_shape, nb_classes,3,weight)
       #self.train_images, self.test_images,self.train_labels,self.test_labels=train_test_split(self.train_images, self.train_labels,train_size=0.30 ,test_size=0.20)
       self.logger.info("+++++++++++The dataset has been reade and splited")
     
        
        
   def execute( self,  task_name: str,  shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal, ) -> Shareable:
       
    
        # retrieve model weights download from server's shareable
       if abort_signal.triggered:
           return make_reply(ReturnCode.TASK_ABORTED)

       if task_name != "train":
           return make_reply(ReturnCode.TASK_UNKNOWN)
       if task_name == self._train_task_name:
          # train_model = ConvNet()
           #self.train_images, self.test_images,self.train_labels,self.test_labels=train_test_split(self.train_images, self.train_labels,train_size=0.7 ,test_size=0.7)
           if not os.path.isdir('%s_model_' %(self.client_name)):
               weight = []
               print(len(self.train_images),len(self.train_images))
               flag=False
               model,train_model = _trainModel(self.train_images, self.test_images,self.train_labels,self.test_labels,self.input_shape, self.nb_classes,3,weight,flag)
              # model = train_model.network_fcN(self.input_shape,self.nb_classes,weight,flag=False)                       
              # train_model.trainNet(self.model,self.train_images,self.train_labels,self.train_labels,16,self.epochs_per_round)
               model.save('%s_model_' %(self.client_name))    
               self.logger.info("+++++++++++model is created")
           else:           
                model_old = keras.models.load_model('%s_model_' %(self.client_name))
                weights_old=model_old.weights
                flag=True
                model,train_model = _trainModel(self.train_images, self.test_images,self.train_labels,self.test_labels,self.input_shape, self.nb_classes,3,weights_old,flag)
              
                self.logger.info("+++++++++++model is loaded")
           
            
           
           
          
           print(len(self.train_images),len(self.test_images))
           dxo = from_shareable(shareable)
          # print(fl_ctx.get_prop(AppConstants.CURRENT_ROUND))
           if 'global_Graph' in dxo.data:
               self.log_info(fl_ctx, "***********new graph has beeen loaded")
               self._global_graph.append(dxo.data['global_Graph'])
               self._centriod.append(dxo.data['contriod'])
               print(dxo.data['global_Graph'])
               print(dxo.data['global_Graph'].nodes())
               
              
           else:
              
               self.log_info(fl_ctx, "***********initialization weigths are recived from the server")
               global_weights=dxo.data
               model.set_weights(global_weights)
               self.model = model
           
          
    
           
           # # train network
          
           # start_time = time.time()
           # ##should be taken out later 
           # print("train FCN model %s seconds ---" % (time.time() - start_time))
           
           #  #extract the MHAP
           # start_time = time.time()
           visulization_traning = HighlyActivated(self.model,train_model,self.train_images,self.train_images,self.nb_classes,netLayers=3)
           self.log_info(fl_ctx, "*********** HighlyActivated done")
         
           activation_layers = visulization_traning.Activated_filters(example_id=1)
           self.log_info(fl_ctx, "*********** visulization_traning done")
           period_active,threshold = visulization_traning.get_index_MHAP(activation_layers,kernal_size=[8,5,3])
           self.log_info(fl_ctx, "*********** get_index_MHAP done")
           # print("extract MHAP %s seconds ---" % (time.time() - start_time))
           # start_time = time.time()           
           
            #clustering the extracted MHAP
           cluster_central = _clusering_array(period_active,self.numRound)
           print('done clusterning')
           # print("clusterning %s seconds ---" % (time.time() - start_time))
           # start_time = time.time()
             #build the graph
           G,sample_cluster_mhap = visulization_traning.get_graph_MHAP(activation_layers,[8,40,120],cluster_central,threshold,6,10)
           # print("get the graph %s seconds ---" % (time.time() - start_time))           
           print('segment')
      
           

           if not (self.numRound == 6 or self.numRound  == 11 or self.numRound ==19 or self.numRound == 0):
                self.log_info(fl_ctx, "***********local aggregation will run in this round")
           
                self._global_graph.append(G)
                self._centriod.append(cluster_central)
                print(len(self._centriod[0][0]))
                print(len(self._centriod[1][0]))
                #print(G)
                #print(G.nodes)
                #print(self._global_graph)
                pickle.dump(self._global_graph, open("graphs_.pkl", "wb"))
                pickle.dump(self._centriod, open("centeriod.pkl", "wb"))

                G,cluster_central = _server_graph_aggregation(self._global_graph,self._centriod)
           
           
           print(flag)
           # start_time = time.time()
           name_sample = '%s_sample_cluster_mhap.npy' %(fl_ctx.get_identity_name())
           np.save(name_sample,sample_cluster_mhap)
           print('save_segment')
           #name_y = '%s_y.npy' %(fl_ctx.get_identity_name())
           y_train = np.argmax(self.train_labels, axis=1)
           print('Y_true: ',y_train)
           pickle.dump(y_train, open("_y_Train.pkl", "wb"))
           #np.save(name_y,y_train)
          
           model.save('%s_model_' %(self.client_name))    
           
           #report updated weights in shareable
           collection= {'graph':G,'centriod':cluster_central}           
           dxo = DXO(data_kind=DataKind.WEIGHTS, data=collection)
           pickle.dump(cluster_central, open("cluster_central.pkl", "wb"))
           
           #dill.dump_session('globalsave_new.pkl')
           self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
           new_shareable = dxo.to_shareable()
           print('Round Number: ',self.numRound)
           self.numRound+=1
           self._global_graph=[]
           self._centriod=[]
           
           
           return new_shareable
       
        
      
        
       
    