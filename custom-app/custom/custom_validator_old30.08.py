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
        data_name = "Wafer"
        #global_graph=nx.from_dict_of_dicts(global_graph,create_using=nx.DiGraph())
        #print(global_graph.nodes())
        graph_embaded = Graph_embading(global_graph)
        #graph_embaded.drwa_graph()
        node_names = graph_embaded.get_node_list()
        #node_names=[0,3,1,7,2,4]
        #print(node_names)
        walks_nodes = graph_embaded.randome_walk_nodes(node_names)
        embaded_graph = graph_embaded.embed_graph(walks_nodes)
        #graph_embaded.plot_embaded_graph(embaded_graph,node_names)
        name_sample = '%s_sample_cluster_mhap.npy' %(fl_ctx.get_identity_name())
        sample_cluster_mhap = np.load(name_sample,allow_pickle=True)
        self.logger.info('mhap graph is loaded ++++++++++++++++++++++++')
       # print(sample_cluster_mhap)
        new_feature = _timeseries_embedding(embaded_graph,node_names,sample_cluster_mhap,6)
        #print(new_feature)
        x_train_feature = []
        name_y = '%s_y.npy' %(fl_ctx.get_identity_name())
        y_true = np.load(name_y,allow_pickle=True)
        self.logger.info('y_true is loaded ++++++++++++++++++++++++')
        
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
        y_train =y_true
        X_train, X_test, y_train, y_test = train_test_split(x_train_new, y_true, test_size=0.2)
        model = xgb.XGBClassifier()
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        CM = confusion_matrix(y_test, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = TP/(TP+FN) #or recall
        specificity = TN/(FP+TN)
        BalanceACC = (sensitivity+specificity)/2
        G_mean= math.sqrt(sensitivity*specificity)
        FN_rate= FN/(FN+TP) 
        FP_rate = FP/(FP+TN)
        Precision = TP/(TP+FP)
        F1_score=2 * (Precision * sensitivity) / (Precision + sensitivity)
        #model = xgb.XGBClassifier()
        self.logger.info('evaliatioion is started ++++++++++++++++++++++++')
        # evaluate the model
        #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        # print(x_train_new)
        # print(y_true)
        #n_scores = cross_val_score(model, x_train_new, y_true, scoring='accuracy', cv=cv, n_jobs=-1)
        com_round=shareable.get_header(AppConstants.CONTRIBUTION_ROUND)
        #print('Client %s ,Communication round %s, Accuracy: %.3f (%.3f)' % (fl_ctx.get_identity_name(),com_round,mean(n_scores), std(n_scores)))
        print('Client %s ,Communication round %s, Accuracy: %.3f' % (fl_ctx.get_identity_name(),com_round,accuracy))
        #res=mean(n_scores)
        #here we save the data in a file
        name = '%s_graph_accuracy.csv' %(data_name)
        results_data = [accuracy,sensitivity,specificity,BalanceACC,G_mean,FN_rate,FP_rate,Precision,F1_score]
        with open(r'%s'%(name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(results_data)
        dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": accuracy,"TP":TP,"TN":TN,"FP":FP,"FN":FN})
        
        return dxo.to_shareable()
     
