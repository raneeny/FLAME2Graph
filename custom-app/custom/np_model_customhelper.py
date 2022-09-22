import re
from typing import Optional

##############################################################
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
import dill
import networkx as nx
      ##############################################################
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
               simi_client.append(_similarity_array(layers,centriods[i][k]))
           clients_sim.append(simi_client)
           i +=1
       #print(clients_sim)
       return clients_sim

def _relabiling_client_graph(graph,maping):  
        mapping_graph = {}
        node_name = list(graph.nodes)
        for i in node_name:
            new_name = maping[int(i[5])][int(i[7])]
            new_name = 'layer%s %s' %(i[5],new_name)
            mapping_graph[i] = new_name
        G = nx.relabel_nodes(graph, mapping_graph)
        return G

def _server_graph_aggregation(graphs,centriods):
         #find the node similarity between all the clients local graphs
         #find the node similarity between all the clients local graphs
        new_cent_index = _similirity_node_name(centriods)
        #relabiling the graphs based on the new similarity
        new_graph = []
        for i in range(len(graphs)):
            new_graph.append(_relabiling_client_graph(graphs[i],new_cent_index[i]))
        #comose the new clients graph into a new one
        Graph1 = new_graph[0]
        i = 1
        global_graph=nx.Graph()
        while i < len(new_graph):
            global_graph = nx.compose(Graph1,new_graph[i])
            i+=1
        return global_graph,centriods[0]
      ######################################################################## 
          
      ######################################################################## 

class CustomHelper(object):
    def __init__(self):
        """Perform weighted aggregation.

        Args:
            exclude_vars (str, optional): regex string to match excluded vars during aggregation. Defaults to None.
        """
        super().__init__()
       
        self.reset_stats()
        self.graphs = list()
        self.centriods=list()
        self.counts = dict()
        self.history = list()
        self.weights=list()


    def reset_stats(self):
        self.graphs = []
        self.counts = {}
        self.centriods=[]
        self.history = []
        self.weights=[]



    def add_data(self, collection, contributor_name, contribution_round):
        graph=collection['graph']
        centriod=collection['centriod']
        print(type(centriod))
        #model_weights=collection['weights']
        #graph=nx.from_dict_of_dicts(graph,create_using=nx.DiGraph())
        if centriod is None or graph is None: 
            print('Gtaph or Centeriod are empty arrays')
        self.graphs.append(graph)
        self.centriods.append(centriod)
        #self.weights.append(model_weights)
        self.history.append(
            {
                "contributor_name": contributor_name,
                "round": contribution_round,
            }
        )
        



    def get_global_graph(self):
      
        aggregated_graph,contriod_c = _server_graph_aggregation(self.graphs,self.centriods)
        #print(aggregated_graph.nodes())
        if aggregated_graph is not None:
                dill.dump_session('finalGraph.pkl')

        else:
            self.logger.info("###############resulted Graph is None")
        self.reset_stats()
        return aggregated_graph,contriod_c



    def get_history(self):
        return self.history



    def get_len(self):
        return len(self.get_history())