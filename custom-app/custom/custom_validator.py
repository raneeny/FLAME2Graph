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
from sklearn.metrics import confusion_matrix
from numpy import mean
from numpy import std
import networkx as nx
np.random.seed(0)
import pathlib
import platform
import pickle
import math,csv
###################################flare imports####################

from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score,precision_score


from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants





#define function of Time Series Embedding
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

class SimpleValidator(Executor):
    
    
   def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(SimpleValidator,self).__init__()
        self._validate_task_name = validate_task_name
        self.num_round=0
        self.logger.info("+++++++++++Iniliaastion of simple validator")
       

        
   def execute( self,  task_name: str,  shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal, ) -> Shareable:
        
        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
           return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "validate" :
           return make_reply(ReturnCode.TASK_UNKNOWN)
      
        self.logger.info('testing phase is started ++++++++++++++++++++++++')
        dxo = from_shareable(shareable)
        global_graph = dxo.data['global_Graph']
        data_name = "PAMAP2"
        name_x = '%s_X_test.npy' %(fl_ctx.get_identity_name())
        name_y = '%s_Y_test.npy' %(fl_ctx.get_identity_name())
        x_test = np.load(name_x,allow_pickle=True)
        y_test = np.load(name_y,allow_pickle=True)
        #name_sample = '%s_cluster_central.npy' %(fl_ctx.get_identity_name())
        #cluster_central = np.load(name_sample,allow_pickle=True)
        #cluster_central= pickle.load(open("cluster_central.pkl", "wb"))
        file = open("cluster_central.pkl",'rb')
        cluster_central = pickle.load(file)
        file.close()
        model = keras.models.load_model('%s_model_' %(fl_ctx.get_identity_name()))
      
        
      
        self.logger.info("+++++++++++model is loaded")
        
        #global_graph=nx.from_dict_of_dicts(global_graph,create_using=nx.DiGraph())
        #print(global_graph.nodes())
        graph_embaded = Graph_embading(global_graph)
        #graph_embaded.drwa_graph()
        node_names = graph_embaded.get_node_list()
        #node_names=[0,3,1,7,2,4]
        #print(node_names)
        walks_nodes = graph_embaded.randome_walk_nodes(node_names)
        embaded_graph = graph_embaded.embed_graph(walks_nodes,self.num_round)
        nb_classes=8
        train_model=[]
        visulization_traning = HighlyActivated(model,train_model,x_test,y_test,nb_classes,netLayers=3)
        activation_layers = visulization_traning.Activated_filters(example_id=1)
        period_active,threshold = visulization_traning.get_index_MHAP(activation_layers,kernal_size=[8,5,3])
        g_l,sample_cluster_mhap_test = visulization_traning.get_graph_MHAP(activation_layers,[8,40,120],cluster_central,threshold,6,10)
        y_test = np.argmax(y_test, axis=1)
      
        
        
        #graph_embaded.plot_embaded_graph(embaded_graph,node_names)
        #name_sample = '%s_sample_cluster_mhap.npy' %(fl_ctx.get_identity_name())
        #sample_cluster_mhap = np.load(name_sample,allow_pickle=True)
      
      
        #name_y = '%s_y.npy' %(fl_ctx.get_identity_name())
        #y_true = np.load(name_y,allow_pickle=True)
        #pickle.dump(cluster_central, open("_y_Train.pkl", "wb"))
       # file = open("_y_Train.pkl",'rb')
        #y_true = pickle.load(file)
        #file.close()
        #print('Y_true After: ',y_true)
        #self.logger.info('mhap graph & y_true are loaded ++++++++++++++++++++++++')
        #sample_cluster_mhap = np.concatenate((sample_cluster_mhap, sample_cluster_mhap_test), axis=0)
        #y_true = np.concatenate((y_true, y_test), axis=0)
        sample_cluster_mhap = sample_cluster_mhap_test
        y_true=y_test
       # print(sample_cluster_mhap)
        new_feature = _timeseries_embedding(embaded_graph,node_names,sample_cluster_mhap,6)
        #print(new_feature)
        x_train_feature = []
        
        for m,data in enumerate (new_feature):
            segmant = []
            for j,seg in enumerate(data):
                segmant.append(seg[0])
            x_train_feature.append(segmant)
     
        x_train_new = []
        for i, data in enumerate (x_train_feature):
            seg = []
            for j in (data):
                for k in j:
                    seg.append(k)
            x_train_new.append(seg)
        #y_train =y_true
        X_train, X_test, y_train, y_test = train_test_split(x_train_new, y_true, test_size=0.1)
        
        model = xgb.XGBClassifier()
        # evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        #print(len(x_train_new))
        #print(len(y_true))
        n_scores = cross_val_score(model, x_train_new, y_true, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        #print('Client %s ,Communication round %s, Accuracy: %.3f (%.3f)' % (client_name,com_round,mean(n_scores), std(n_scores)))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        balance_acc = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1_sc = f1_score(y_test, y_pred, average='weighted')
        pres_val = precision_score(y_test, y_pred, average='weighted')
        #model = xgb.XGBClassifier()
        self.logger.info('evaliatioion is started ++++++++++++++++++++++++')
        # evaluate the model
        #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        # print(x_train_new)
        # print(y_true)
        #n_scores = cross_val_score(model, x_train_new, y_true, scoring='accuracy', cv=cv, n_jobs=-1)
        com_round=shareable.get_header(AppConstants.CONTRIBUTION_ROUND)
        #print('Client %s ,Communication round %s, Accuracy: %.3f (%.3f)' % (fl_ctx.get_identity_name(),com_round,mean(n_scores), std(n_scores)))
        print('Client %s ,Communication round %s, Accuracy: %.3f' % (fl_ctx.get_identity_name(),self.num_round,accuracy))
        #res=mean(n_scores)
        #here we save the data in a file
        name = '%s_graph_accuracy.csv' %(data_name)
        results_data = [np.mean(n_scores),balance_acc,f1_sc,pres_val]
        with open(r'%s'%(name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(results_data)
        dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": np.mean(n_scores)})
        self.num_round+=1
        return dxo.to_shareable()
     
