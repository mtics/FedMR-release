# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
import os
import pickle
import platform
from logging import getLogger

from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader, FederatedDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import init_seed, get_model, get_trainer, dict2str, get_combinations, save_experiment_results, \
    find_best_parameters, mail_notice


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info(f'██ Directory: {os.getcwd()} on Server: {platform.node()} ██')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info('>>> [{} stats] Overall: '.format(config['dataset']) + str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('>>> [{} stats] Train:   '.format(config['dataset']) + str(train_dataset))
    logger.info('>>> [{} stats] Valid:   '.format(config['dataset']) + str(valid_dataset))
    logger.info('>>> [{} stats] Test:    '.format(config['dataset']) + str(test_dataset))

    # wrap into dataloader
    if config['is_federated']:
        train_data = FederatedDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        (valid_data, test_data) = (
            FederatedDataLoader(config, valid_dataset, additional_dataset=train_dataset, stage='valid',
                                batch_size=config['eval_batch_size']),
            FederatedDataLoader(config, test_dataset, additional_dataset=train_dataset, stage='test',
                                batch_size=config['eval_batch_size'])
        )

    else:
        train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        (valid_data, test_data) = (
            EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset,
                           batch_size=config['eval_batch_size']),
            EvalDataLoader(config, test_dataset, additional_dataset=train_dataset,
                           batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('=' * 25 + ' Combining Hyper-Parameters ' + '=' * 25)

    # get all combinations of hyper-parameters
    combinators, total_loops = get_combinations(config, config['result_file_name'])

    # loop through all hyper-parameter combinations to find the best
    for hyper_tuple in combinators:

        hyper_dict = {}
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
            hyper_dict[j] = k

        # random seed reset
        init_seed(config['seed'])

        logger.info('=' * 15 + ' [{}/{} Parameter Combination] {}={} '.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple
        ) + '=' * 15)

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer(config['model'], config['is_federated'])(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data,
                                                                                test_data=test_data,
                                                                                saved=save_model)

        if config['save_model']:
            model_dir = config['model_dir'].format(config['type'], config['comment'])
            save_params = {
                'client_models': trainer.client_models,
                'fusion': trainer.fusion,
                'item_commonality': trainer.item_commonality,
                't_feat': trainer.t_feat,
                'v_feat': trainer.v_feat,
                'train_loss': trainer.train_loss_dict,
            }
            with open(model_dir, 'wb') as f:
                pickle.dump(save_params, f)

        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('[Best Valid]: {}'.format(dict2str(best_valid_result)))
        logger.info('[Best Test]:  {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████ Current BEST ████:\nParameters: {}={},\n'
                    'Valid: {},\nTest:  {}\n'.format(config['hyper_parameters'],
                                                     hyper_ret[best_test_idx][0],
                                                     dict2str(hyper_ret[best_test_idx][1]),
                                                     dict2str(hyper_ret[best_test_idx][2])))

        # save the results of the current hyper-parameter combination
        save_experiment_results(hyper_dict, best_test_upon_valid, config['result_file_name'])
        logger.info('Results saved to {}'.format(config['result_file_name']))

    # log info
    logger.info('============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\nBest Valid: {},\nBest Test:  {}'.format(config['hyper_parameters'],
                                                                                      p, dict2str(k), dict2str(v)))

    logger.info(f'█████████████ {config["model"]} on {config["dataset"]} - BEST COMBINATION ████████████████')
    logger.info('\tParameters: {}={},\nValid: \t{},\nTest: \t{}\n\n'.format(config['hyper_parameters'],
                                                                            hyper_ret[best_test_idx][0],
                                                                            dict2str(hyper_ret[best_test_idx][1]),
                                                                            dict2str(hyper_ret[best_test_idx][2])))

    # find the best hyper-parameter combination
    best_result = find_best_parameters(config['result_file_name'], metric=config['valid_metric'].lower())
    notice_info = 'Best hyper-parameter combination: {}'.format(dict2str(best_result))

    if config['notice']:
        mail_notice(config, notice_info)
        logger.info('Sending notice email success!')

    logger.info(notice_info)
