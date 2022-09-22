

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import ReturnCode


from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from typing import Any, Dict, Union, List

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
from nvflare.app_common.app_constant import AppConstants

####################our import ####################
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
          mini = distance.euclidean(data,cluster[0])
          cluster_id = 0
          count = 0
          for i in (cluster):
              clu_nor = _normilization(i)
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
          return clients_sim


def _relabiling_client_graph(graph,maping):  
          mapping_graph = {}
          node_name = list(graph.nodes)
          for i in node_name:
              new_name = maping[int(i[5])][int(i[7])]
              new_name = '%s %s' %(i[5],new_name)
              mapping_graph[i] = new_name
          G = nx.relabel_nodes(graph, mapping_graph)
          return G

def _server_graph_aggregation(graphs,centriods):
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
          return global_graph
          
      ######################################################################## 




class ModelAggregator(Aggregator):
    
    def __init__(
        self,
        #exclude_vars: Union[str, Dict[str, str], None] = None,
        aggregation_weights: Union[Dict[str, Dict[str,any]],None] = None,
        expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHTS,
        
    ):
        super().__init__()
        print('++++++++ aggregator iniliazation+++++++++++++++')
       
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")
        self.logger.debug(f"expected data kind: {expected_data_kind}")

          # Set up DXO aggregators
     #   self.expected_data_kind = expected_data_kind
      
        self._single_dxo_key = ""
        
        aggregation_weights = aggregation_weights or {}
        #self.aggregation_weights= aggregation_weights
        # Check expected data kind
        if isinstance(expected_data_kind, dict):
           for k, v in expected_data_kind.items():
               if v not in [DataKind.WEIGHTS]:
                   raise ValueError(
                       f"expected_data_kind[{k}] = {v} is not {DataKind.WEIGHTS}"
                   )
           self.expected_data_kind = expected_data_kind
        else:
           if expected_data_kind not in [DataKind.WEIGHTS]:
               raise ValueError(
                   f"expected_data_kind = {expected_data_kind} is not {DataKind.WEIGHTS}"
               )
           self.expected_data_kind = {self._single_dxo_key: expected_data_kind}
        self.global_data=dict()
        print(expected_data_kind)
        print(type(expected_data_kind))
        
        
        
        
        
        
        self.global_graph=[]
        self.dxo_aggregators = dict()
        #print(self.aggregation_weights)
        for k in self.expected_data_kind.keys():
           self.dxo_aggregators.update(
               {
                   k: DXOAggregator(
                       
                    #   aggregation_weights=self.aggregation_weights[k],
                       expected_data_kind=self.expected_data_kind[k],
                       name_postfix=k,
                   )
               }
           )
        
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
            
            try:
                dxo = from_shareable(shareable)
                #print(dxo.data)
            except BaseException:
                self.log_exception(fl_ctx, "shareable data is not a valid DXO")
                return False
       
            contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
            contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)

            rc = shareable.get_return_code()
            if rc and rc != ReturnCode.OK:
                self.log_warning(fl_ctx, f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.")
                return False
            n_accepted = 0
            centriod_=[]
            graph_=[]
           
           
          #  print(self.expected_data_kind.keys())
            for key in self.expected_data_kind.keys():
               if key == self._single_dxo_key:  # expecting a single DXO
                   sub_dxo = dxo
               else:  # expecting a collection of DXOs
                   sub_dxo = dxo.data
               self.global_data.update({key:dxo})
  
               graph=dxo.data['graph']
            
               centriod=dxo.data['centriod']
            
               centriod_.append(list(centriod.items()))
               #print(centriod_)
             
             #  print(centriod)
               graph_.append(nx.from_dict_of_dicts(graph,create_using=nx.DiGraph()))
               #print(graph_)
               aggregated_dxo = _server_graph_aggregation(graph_,centriod_)
               print('########################################')
               print(aggregated_dxo)
               self.global_graph.append(aggregated_dxo)
               
               
               
               if not isinstance(sub_dxo, DXO):
                   self.log_warning(fl_ctx, f"Collection does not contain DXO for key {key} but {type(sub_dxo)}.")
                   continue
               # print(self.dxo_aggregators_graph)
               # print(type(self.dxo_aggregators_graph))
               #self.dxo_aggregators[key].aggregation_weights=sub_dxo
              # print(sub_dxo,'##################################################################')
              # accepted = self.dxo_aggregators[key].accept(
               #    dxo=sub_dxo, contributor_name=contributor_name, contribution_round=contribution_round, fl_ctx=fl_ctx  )
               
               
            #    if not accepted:
            #        print('Did note accepcpted ############')
            #        return False
            #    else:
            #        n_accepted += 1
    
            # if n_accepted > 0:
            #    return True
            # else:
            #    self.log_warning(fl_ctx, f"Did not accept any DXOs from {contributor_name} in round {contribution_round}!")
            #    print('Did note accepcpted ############')
               return True
                
                
        
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
            """Perform the aggregation for all the received Shareable from the clients.
            
            Args:
                fl_ctx: FLContext
            
            Returns:
                shareable
            """
            self.log_debug(fl_ctx, "Start aggregation using my custom methode")
            result_dxo_dict = dict()
            # print('######################################')
            # print(self.dxo_aggregators[''].aggregation_weights)
            centriod_=[]
            graph_=[]
            for key in self.expected_data_kind.keys():
                collection= self.global_data[key]
                graph=collection.data['graph']
                centriod=collection.data['centriod']
                centriod_.append(list(centriod.items()))
              #  print(centriod)
                graph_.append(nx.from_dict_of_dicts(graph,create_using=nx.DiGraph()))
                print(self.global_graph)
                G= nx.to_dict_of_dicts(self.global_graph[0],nodelist=self.global_graph[0].nodes)
                aggregated_dxo = DXO(data_kind=DataKind.WEIGHTS, data=G)
                
                
                
                if key == self._single_dxo_key:  # return single DXO with aggregation results
                    return aggregated_dxo.to_shareable()
                self.log_info(fl_ctx, f"Aggregated contributions matching key '{key}'.")
                result_dxo_dict.update({key: aggregated_dxo})
        # return collection of DXOs with aggregation results
            collection_dxo = DXO(data_kind=DataKind.WEIGHTS, data=result_dxo_dict)
            return collection_dxo.to_shareable()
    
     
 