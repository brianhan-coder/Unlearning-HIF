import argparse

from Image.main_image import main as imagemain
from Graph.main_graph import main as graphmain

import logging

def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0, help='gpu?')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--unlearn_ratio', type=float, default=0.01)
    parser.add_argument('--unlearn_iteration', type=int, default=10)
    parser.add_argument('--damp', type=float, default=0.0)
    parser.add_argument('--scale', type=int, default=500)
    parser.add_argument('--numda', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='IF',
                        choices=["HSIC", "IF", "GIF", "Retrain", "Sharding"])
    parser.add_argument('--model_name', type=str, default='simple_cnn',
                        choices=["simple_cnn", "resnet", "vgg", "GCN", "GAT", "GIN", "SAGE", "SGC"])
    parser.add_argument('--dataset_name', type=str, default='mnist',
                        choices=["mnist", "cifar10", "cora", "citeseer", "pubmed", "CS", "Physics", "ogbn-arxiv"])
    parser.add_argument('--exp_type', type=str, default='image', choices=["image", "graph"])

<<<<<<< HEAD
=======

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--exp', type=str, default='Unlearn', choices=["Unlearn", "Attack"])
    # parser.add_argument('--method', type=str, default='GIF', choices=["GIF", "Retrain", "IF", "HSIC"])

    ########################## unlearning task parameters ######################
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics", "ogbn-arxiv"])
    # parser.add_argument('--unlearn_task', type=str, default='edge', choices=["edge", "node", 'feature'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)

    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_retrain', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=False)
    parser.add_argument('--is_use_batch', type=str2bool, default=True, help="Use batch train GNN models.")
    parser.add_argument('--target_model', type=str, default='GAT', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN","SGC"])
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)

    ########################## GIF parameters ###########################
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--scale', type=int, default=500)
    parser.add_argument('--damp', type=float, default=0.0)

>>>>>>> 91def5f (first push)
    args = vars(parser.parse_args())

    return args

def main():
    args = parameter_parser()

    logger_name = "_".join((args['dataset_name'], str(args['unlearn_ratio']), args['model_name'], args['method'],
                            str(args['unlearn_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    if args["exp_type"].lower() == "image":
        imagemain(args)
    if args["exp_type"].lower() == "graph":
        for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            args['k_ratio'] = i

            _set_random_seed(20221012)
            graphmain(args)

if __name__ == "__main__":
    main()








