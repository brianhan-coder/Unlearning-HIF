<<<<<<< HEAD
import os
import torch

=======
import logging
import os
import torch
import sys
import numpy as np
import random

from GNN.exp.exp_GIF import GraphExpGraphInfluenceFunction
from GNN.exp.exp_retrain import GraphExpRetraining


def _set_random_seed(seed=2022):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')


def main(args)
    args = args

    _set_random_seed(20231004)

    # config the logger
    logger_name = "_".join((args['dataset_name'], str(args['test_ratio']), args['target_model'], args['unlearn_task'],
                            str(args['unlearn_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args["exp"].lower() == "unlearn":
        if args["method"].lower() == "retrain":
            GraphExpRetraining(args)
        elif args["method"].lower() in ["gif", "if", "hsic"]:
            GraphExpGraphInfluenceFunction(args)
        else:
            raise NotImplementedError

    return 0
>>>>>>> 91def5f (first push)
