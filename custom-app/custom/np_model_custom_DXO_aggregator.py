import logging
from typing import Any, Dict, Optional

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from np_model_customhelper import CustomHelper
from nvflare.app_common.app_constant import AppConstants

import networkx as nx

class CustomDXOAggregator(FLComponent):
    def __init__(
        self,
        expected_data_kind: DataKind = DataKind.WEIGHTS,
        name_postfix: str = "",
    ):
        super().__init__()
        self.expected_data_kind = expected_data_kind
        
        self.aggregation_helper = CustomHelper()
        
        self.warning_count = {}
        self.warning_limit = 10

        
    def reset_aggregation_helper(self):
        if self.aggregation_helper:
            self.aggregation_helper.reset_stats()
            
        
            
    def accept(self, dxo: DXO, contributor_name, contribution_round, fl_ctx: FLContext) -> bool:


        if not isinstance(dxo, DXO):
            self.log_error(fl_ctx, f"Expected DXO but got {type(dxo)}")
            return False

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS):
            self.log_error(fl_ctx, "cannot handle data kind {}".format(dxo.data_kind))
            return False

        if dxo.data_kind != self.expected_data_kind:
            self.log_error(fl_ctx, "expected {} but got {}".format(self.expected_data_kind, dxo.data_kind))
            return False

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            self.log_error(fl_ctx, f"unable to accept DXO processed by {processed_algorithm}")
            return False

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_debug(fl_ctx, f"current_round: {current_round}")

        data = dxo.data
        if data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False

        n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"discarding DXO from {contributor_name} at round: "
                f"{contribution_round}. Current round is: {current_round}",
            )
            return False

        for item in self.aggregation_helper.get_history():
            if contributor_name == item["contributor_name"]:
                prev_round = item["round"]
                self.log_warning(
                    fl_ctx,
                    f"discarding DXO from {contributor_name} at round: "
                    f"{contribution_round} as {prev_round} accepted already",
                )
                return False

        if n_iter is None:
            if self.warning_count.get(contributor_name, 0) <= self.warning_limit:
                self.log_warning(
                    fl_ctx,
                    f"NUM_STEPS_CURRENT_ROUND missing in meta of DXO"
                    f" from {contributor_name} and set to default value, 1.0. "
                    f" This kind of message will show {self.warning_limit} times at most.",
                )
                if contributor_name in self.warning_count:
                    self.warning_count[contributor_name] = self.warning_count[contributor_name] + 1
                else:
                    self.warning_count[contributor_name] = 0
            n_iter = 1.0
       
       
      


        # aggregate
        self.aggregation_helper.add_data(data , contributor_name, contribution_round)
        self.log_debug(fl_ctx, "End accept")
        return True
    
    
    
    def aggregate(self, fl_ctx: FLContext) -> DXO:
      

       self.log_debug(fl_ctx, "Start aggregation using custom aggregator")
       current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
       self.log_info(fl_ctx, f"aggregating {self.aggregation_helper.get_len()} update(s) at round {current_round}")
       self.log_debug(fl_ctx, f"complete history {self.aggregation_helper.get_len()}")
       aggregated_graph,contriod_c  = self.aggregation_helper.get_global_graph()
       #print(aggregated_graph)
       
       self.log_debug(fl_ctx, "End aggregation")
       #aggregated_graph= nx.to_dict_of_dicts(aggregated_graph,nodelist=aggregated_graph.nodes)
       collection={'global_Graph':aggregated_graph,'contriod':contriod_c}
       dxo = DXO(data_kind=self.expected_data_kind, data=collection)
       return dxo