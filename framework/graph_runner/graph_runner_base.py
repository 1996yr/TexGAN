import os
import tensorflow as tf
import importlib
from exp_config.config import Config

from ..utils.log import init_logger, log_message


class graph_runner:
    def __init__(self):
        self.submodule_dict = {}
        self.graph = None
        self.sess = None
        self.skip_iter = 0 # training from scratch or previous iteration
        os.makedirs(Config.global_cfg.folder_cfg.output_dir, exist_ok=True)
        os.makedirs(Config.global_cfg.folder_cfg.log_dir, exist_ok=True)

        os.makedirs(Config.global_cfg.folder_cfg.validation_dir, exist_ok=True)
        os.makedirs(Config.global_cfg.folder_cfg.model_dir, exist_ok=True)
        os.makedirs(Config.global_cfg.folder_cfg.dense_gen_dir, exist_ok=True)
        os.makedirs(Config.global_cfg.folder_cfg.test_result_dir, exist_ok=True)

        if Config.global_cfg.folder_cfg.temp_output_dir != '': # for philly temp local storage
            os.makedirs(Config.global_cfg.folder_cfg.temp_output_dir, exist_ok=True)
            os.makedirs(Config.global_cfg.folder_cfg.temp_validation_dir, exist_ok=True)
            os.makedirs(Config.global_cfg.folder_cfg.temp_dense_gen_dir, exist_ok=True)

        log_file = os.path.join(Config.global_cfg.folder_cfg.log_dir, 'log.txt')
        init_logger(log_file)
        log_message('graph_runner', '------Using GPU: {}------'.format(Config.global_cfg.meta_cfg.gpu_list))  # example [1, 2]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            list(map(str, Config.global_cfg.meta_cfg.gpu_list)))  # example [1,2] -> ['1', '2'] -> '1,2'

    def build_graph(self):
        if self.graph is not None:
            del self.graph
        self.graph = tf.Graph()

    def init_session(self):
        with self.graph.as_default():
            tf.set_random_seed(0)
            self.saver = tf.train.Saver(max_to_keep=Config.global_cfg.meta_cfg.max_num_checkpoint)
            self.init_op_global = tf.global_variables_initializer()
            self.init_op_local = tf.local_variables_initializer()
        log_message(self.__class__.__name__, '-----Initialize Tensorflow...-----')
        config_pro = tf.ConfigProto()
        config_pro.allow_soft_placement = True
        config_pro.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_pro, graph=self.graph)
        self.tensorboard_writer = tf.summary.FileWriter(Config.global_cfg.folder_cfg.log_dir, self.sess.graph)

        log_message(self.__class__.__name__, '-----Initialize variables...-----')
        self.sess.run(self.init_op_global)
        self.sess.run(self.init_op_local)

    def load_previous_model(self, model=''):
        _loaded = False
        if model != '':
            self.saver.restore(self.sess, model)
            log_message(self.__class__.__name__, '---Loaded previous model {}...----'.format(model))
            _loaded = True
        else:
            if os.path.exists(Config.global_cfg.folder_cfg.previous_model + '.meta'):
                log_message(self.__class__.__name__, '-----Load previous model (if have)...-----')
                self.saver.restore(self.sess, Config.global_cfg.folder_cfg.previous_model)
                log_message(self.__class__.__name__, '-----Restored {}...-----'.format(Config.global_cfg.folder_cfg.previous_model))
                _loaded = True
            elif Config.global_cfg.meta_cfg.auto_restart:
                latest_checkpoint = tf.train.latest_checkpoint(Config.global_cfg.folder_cfg.model_dir)
                if latest_checkpoint != None:
                    checkpoint_iter = int(latest_checkpoint.split('-')[-1])
                    self.saver.restore(self.sess, latest_checkpoint)
                    log_message(self.__class__.__name__, '-----Auto restored {}...-----'.format(latest_checkpoint))
                    log_message(self.__class__.__name__, '-----Training begin from iter {}...-----'.format(checkpoint_iter))
                    self.skip_iter = checkpoint_iter
                    _loaded = True
                else:
                    log_message(self.__class__.__name__, '-----No previous model found.-----')
            else:
                log_message(self.__class__.__name__, '-----No previous model found.-----')

        return _loaded

    def close_session(self):
        if self.sess is not None:
            self.sess.close()
        self.sess = None