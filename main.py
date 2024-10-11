# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MMFedRAP', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='KU', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='set the gpu id')
    parser.add_argument('--type', '-t', type=str, default='default', help='variant of the type')
    parser.add_argument('--comment', '-c', type=str, default='default', help='comment of the experiment')


    args, _ = parser.parse_known_args()

    config_dict = {}

    # 将args转为dict，更新到config_dict中
    config_dict.update(vars(args))

    return config_dict, args



if __name__ == '__main__':

    config_dict,args = load_config()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)
