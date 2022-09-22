import os
import pickle
import json


from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.model import make_model_learnable
from ConvNet_Model import ConvNet

class CustomModelPersistor(ModelPersistor):
    def __init__(self, save_name="_model.pkl"):
        super().__init__()
        self.save_name = save_name

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        #fl_ctx.set_prop(AppConstants.CURRENT_ROUND,0)
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        env = None
        print(app_root,'*****************')
        run_args = fl_ctx.get_prop(FLContextKey.ARGS)
        if run_args:
            env_config_file_name = os.path.join(app_root, run_args.env)
            if os.path.exists(env_config_file_name):
                try:
                    with open(env_config_file_name) as file:
                        env = json.load(file)
                     #   self.logger.info(f"load the model++++++++++++++++++++++")
                        
                except:
                    self.system_panic(
                        reason="error opening env config file {}".format(env_config_file_name), fl_ctx=fl_ctx
                    )
                    return

        if env is not None:
            if env.get("APP_CKPT_DIR", None):
                fl_ctx.set_prop(AppConstants.LOG_DIR, env["APP_CKPT_DIR"], private=True, sticky=True)
            if env.get("APP_CKPT") is not None:
                fl_ctx.set_prop(
                    AppConstants.CKPT_PRELOAD_PATH,
                    env["APP_CKPT"],
                    private=True,
                    sticky=True,
                )

        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            self.log_dir = app_root
        self._pkl_save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.
        Args:
            fl_ctx: FLContext
        Returns:
            Model object
        """

        # if os.path.exists(self._pkl_save_path):
        #     self.logger.info(f"Loading server weights++++++++++++++++++++++++++++")
        #     with open(self._pkl_save_path, "rb") as f:
        #         model_learnable = pickle.load(f)
        # else:
        self.logger.info(f"Initializing server model+++++++++++++++++++++++++++")
        network =  ConvNet()
        input_shape=(600,17) #198 for wafer
        model = network.network_fcN(input_shape,8,[],False)
      
 
        #model = train_model.network_fcN(self.input_shape,self.nb_classes,weight,flag=False)       
         
        #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #network.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        #_ = network(tf.keras.Input(shape=(28, 28)))
        var_dict={}
        for x,y in enumerate(model.get_weights()):
           var_dict[x]=y
        #var_dict = model.get_weights()
        model_learnable = make_model_learnable(var_dict, dict())
        print('modelll is loded from that cripy function ***************')
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.
        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        model_learnable_info = {k: str(type(v)) for k, v in model_learnable.items()}
        self.logger.info(f"Saving aggregated server graph: \n {model_learnable_info}")
        with open(self._pkl_save_path, "wb") as f:
            pickle.dump(model_learnable, f)