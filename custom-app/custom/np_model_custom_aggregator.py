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
from np_model_custom_DXO_aggregator import CustomDXOAggregator
from nvflare.app_common.app_constant import AppConstants


class ModelAggregator(Aggregator):
    
    def __init__(
        self,
        expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHTS,
        
    ):
        super().__init__()
        print('++++++++ aggregator iniliazation+++++++++++++++')
       
    
        self.logger.debug(f"expected data kind: {expected_data_kind}")

          # Set up DXO aggregators
     #   self.expected_data_kind = expected_data_kind
      
        self._single_dxo_key = ""
        
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
           
           
           
           self.dxo_aggregators = dict()
           #print(self.aggregation_weights)
           for k in self.expected_data_kind.keys():
              self.dxo_aggregators.update(
                  {
                      k: CustomDXOAggregator(
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
                 
                 for key in self.expected_data_kind.keys():
                    if key == self._single_dxo_key:  # expecting a single DXO
                        sub_dxo = dxo
                    else:  # expecting a collection of DXOs
                        sub_dxo = dxo.data
                
                    accepted = self.dxo_aggregators[key].accept(
                        dxo=sub_dxo, contributor_name=contributor_name, contribution_round=contribution_round, fl_ctx=fl_ctx)
                    
                    if not accepted:
                            return False
                    else:
                            n_accepted += 1

                 if n_accepted > 0:
                         return True
                 else:
                         self.log_warning(fl_ctx, f"Did not accept any DXOs from {contributor_name} in round {contribution_round}!")
                         return False
    def aggregate(self, fl_ctx: FLContext) -> Shareable:

     
        result_dxo_dict = dict()
      
        for key in self.expected_data_kind.keys():
            aggregated_dxo = self.dxo_aggregators[key].aggregate(fl_ctx)
            if key == self._single_dxo_key: 
                return aggregated_dxo.to_shareable()
            self.log_info(fl_ctx, f"Aggregated contributions matching key '{key}'.")
            result_dxo_dict.update({key: aggregated_dxo})
           
        # return collection of DXOs with aggregation results
        collection_dxo = DXO(data_kind=DataKind.WEIGHTS, data=result_dxo_dict)
        #print(result_dxo_dict,"#######################################")
        return collection_dxo.to_shareable()