import argparse
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNNs for Time Series Forecasting')

    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PatchConvBiLSTM', help='model name, options:PatchConvBiLSTM')

    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default=r'./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=r'RAL_TCM.csv', help='data file')
    parser.add_argument('--row_id', type=int, default=5, help='row id of the data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target1', type=str, default='TN\L2_AGC_S3TRFC_ACT',
                        help='target feature in S or MS task, options:[TN\L2_AGC_S1TRFC_ACT, '
                             'TN\L2_AGC_S2TRFC_ACT,'
                             'TN\L2_AGC_S3TRFC_ACT,'
                             'TN\L2_AGC_S4TRFC_ACT,'
                             'TN\L2_AGC_S5TRFC_ACT,'
                             'TN\L2_AGC_V2THK_ACT,'
                             'TN\L2_AGC_V5THK_ACT,'
                             'TN\L2_AGC_H5THK_ACT]')
    parser.add_argument('--target2', type=str, default='TN\L2_AGC_S1TRFC_ACT',
                        help='target feature in S or MS task')
    parser.add_argument('--num_predict', type=int, default=1,
                        help='number of steps to predict, if you want to predict all, set 0')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default=r'./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--individual', type=bool, default=False,
                        help='Controls whether the model uses a separate linear layer for each channel when processing the input data')
    parser.add_argument('--sclae', type=str, default=True, help='Whether to perform data scaling')
    parser.add_argument('--csv_path', type=str, default=r'outputs',
                        help='excel data path in order to save some metricx')

    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--patch_len', type=int, default=50, help='patch length')
    parser.add_argument('--pred_len', type=int, default=100,
                        help='prediction sequence length')
    parser.add_argument('--patch_pred_len', type=int, default=50,
                        help='patch_pred length')
    parser.add_argument('--seq_cha', type=int, default=170, help='channel or dimension for sequence')
    parser.add_argument('--enc_in', type=int, default=50, help='channel or dimension for encoder')

    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--weight', default='dwa', type=str, help='choose weighting function')
    parser.add_argument('--temp', default=2, type=int, help='temperature')
    parser.add_argument('--t_dim', type=int, default=1, help='number of predictions')
    parser.add_argument('--disable_rev', action='store_true', help='whether to disable RevIN')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--modes1', type=int, default=64, help='modes to be 64')

    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--dec_in', type=int, default=170, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=170, help='output size')

    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--res_attention', type=bool, default=True, help='res attention')

    parser.add_argument('--label_len', type=int, default=48, help='unused fot this model without Autoformer')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--ab', type=int, default=2, help='ablation version')
    parser.add_argument('--seasonal', type=int, default=7)
    parser.add_argument('--mode_type', type=int, default=0)
    parser.add_argument('--ours', default=False, action='store_true')
    parser.add_argument('--wavelet', type=int, default=0)
    parser.add_argument('--version', type=int, default=16)
    parser.add_argument('--ratio', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=6, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=35, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=120, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_sl_{}_prl_{}_ei_{}_pl_{}_{}_pprl_{}_{}'.format(
                args.target1,
                args.model,
                args.features,
                args.seq_len,
                args.pred_len,
                args.enc_in,
                args.patch_len,
                args.train_epochs,
                args.patch_pred_len,
                ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        setting = 'STD3_M100_50_50_50_mix_std'

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
