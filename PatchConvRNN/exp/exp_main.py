from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchConvBiLSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import warnings
import numpy as np
import pandas as pd
from utils.losses import *

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.num_data = 0
        self.epoch = 0
        self.best_metrics = [1, 0]
        self.stop = 0

    def _build_model(self):
        model_dict = {
            'PatchConvBiLSTM': PatchConvBiLSTM
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == 'test':
            data_set, data_loader, num_data = data_provider(self.args, flag)
            return data_set, data_loader, num_data
        else:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader

    def every_step_evaluation(self, preds, trues, num_data=None, num_feature=None, pred_len=None, setting=None):
        csv_path = self.args.csv_path
        csv_path = os.path.join(csv_path, setting)
        csv_filename1 = 'every_step_evaluation.csv'
        full_csv_path1 = os.path.join(csv_path, csv_filename1)

        col_names = ["MAE", "MSE", "RMSE", "MAPE", "MSPE", "RSE", 'Max_error', 'Median_absolute_error', 'ev_scores',
                     'r2_scores', 'Adjust_R2_scores']
        results_df = pd.DataFrame(columns=col_names)
        for feature_index in range(preds.shape[2]):
            for time_step in range(preds.shape[1]):
                y_pred = preds[:, time_step, feature_index]
                y_true = trues[:, time_step, feature_index]

                mae, mse, rmse, mape, mspe, rse, Max_error, Median_absolute_error, ev_scores, r2_scores, Adjust_R2_scores, _ = metric(
                    pred=y_pred, true=y_true, num_data=num_data, num_feature=num_feature, pred_len=pred_len)

                metrics = [mae, mse, rmse, mape, mspe, rse, Max_error, Median_absolute_error, ev_scores, r2_scores,
                           Adjust_R2_scores]

                temp_df = pd.DataFrame([metrics], columns=col_names, index=[time_step + feature_index * preds.shape[1]])

                results_df = pd.concat([results_df, temp_df])

        results_df.reset_index(drop=True, inplace=True)

        results_df.to_csv(full_csv_path1, mode='w', header=True, index=False)
        return

    def padding_the_data(self, data):
        preds = data
        fill_num = -1e9

        max_first_dim = max([x.shape[0] for x in preds])

        for i in range(len(preds)):
            if preds[i].shape[0] < max_first_dim:
                num_fill_rows = max_first_dim - preds[i].shape[0]

                fill_data = np.full((num_fill_rows, preds[i].shape[1], preds[i].shape[2]), fill_num)

                preds[i] = np.concatenate((preds[i], fill_data), axis=0)

        preds = np.array(preds)
        return preds

    def every_channel_evaluation(self, preds, trues, num_data=None, num_feature=None, pred_len=None, setting=None):
        csv_filename1 = 'every_channel_evaluation.csv'
        csv_path = os.path.join(self.args.csv_path, setting)
        full_csv_path1 = os.path.join(csv_path, csv_filename1)
        metrics_results = pd.DataFrame(
            columns=["MAE", "MSE", "RMSE", "MAPE", "MSPE", "RSE", 'Max_error', 'Median_absolute_error', 'ev_scores',
                     'r2_scores', 'Adjust_R2_scores', 'STD'])

        for i in range(preds.shape[-1]):
            preds_channel = preds[:, :, i]
            true_channel = trues[:, :, i]

            channel_metrics = metric(pred=preds_channel, true=true_channel, num_feature=num_feature, num_data=num_data,
                                     pred_len=pred_len)

            metrics_results.loc[i] = channel_metrics

        metrics_results.to_csv(full_csv_path1, index=False)

        return

    def save_overall_evaluation(self, mae, mse, rmse, mape, mspe, rse, Max_error, Median_absolute_error, ev_scores,
                                r2_scores, Adjust_R2_scores, STD, setting):
        df = pd.DataFrame({
            'MAE': [mae],
            'MSE': [mse],
            'RMSE': [rmse],
            'MAPE': [mape],
            'MSPE': [mspe],
            'RSE': [rse],
            'Max_error': [Max_error],
            'Median_absolute_error': [Median_absolute_error],
            'ev_scores': [ev_scores],
            'r2_scores': [r2_scores],
            'Adjust_R2_scores': [Adjust_R2_scores],
            'STD': [STD]
        })

        csv_path = os.path.join(self.args.csv_path, setting)

        csv_filename1 = 'overall_evaluation.csv'
        full_csv_path1 = os.path.join(csv_path, csv_filename1)

        df.to_csv(full_csv_path1, index=False)
        return

    def AE(self, preds, trues, setting):

        csv_path = os.path.join(self.args.csv_path, setting)

        csv_filename1 = 'Absolute_Error.csv'
        full_csv_path1 = os.path.join(csv_path, csv_filename1)
        combined_data = pd.DataFrame()
        new_columns = 0
        for i in range(preds.shape[2]):
            data_slice = preds[:, :, i]
            if trues is not None:
                true_slice = trues[:, :, i]
                data_slice = abs(data_slice - true_slice)

            df = pd.DataFrame(data_slice)

            new_column_names = [f'Var_{j + 1 + new_columns}' for j in range(df.shape[1])]
            df.columns = new_column_names

            combined_data = pd.concat([combined_data, df], axis=1)

        combined_data.to_csv(full_csv_path1, index=False)
        return

    def save_preds(self, preds, flag=None, setting=None):

        flag = flag
        preds = preds
        row_id = str(self.args.row_id)

        csv_path = os.path.join(self.args.csv_path, setting)
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        if flag == 'preds':
            file_name = f'{csv_path}//test_preds_Y.csv'
        elif flag == 'trues':
            file_name = f'{csv_path}//test_trues_Y.csv'
        else:
            file_name = f'{csv_path}//test_inputs_X.csv'

        combined_data = pd.DataFrame()
        new_columns = 0
        for i in range(preds.shape[2]):
            data_slice = preds[:, :, i]

            df = pd.DataFrame(data_slice)

            new_column_names = [f'Var_{j + 1 + new_columns}' for j in range(df.shape[1])]
            df.columns = new_column_names

            combined_data = pd.concat([combined_data, df], axis=1)

        combined_data.to_csv(file_name, index=False)
        return

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(
        ), lr=self.args.learning_rate, weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        criterion = STD_loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        f_dim = -self.args.num_predict
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for j, batch in enumerate(vali_loader):
                if batch is None:
                    continue
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        if self.args.model == 'MoLE_RLinear' or self.args.model == 'MoLE_DLinear':
                            outputs = self.model(batch_x, batch_x_mark)
                        elif self.args.model == 'Autoformer':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            if self.args.model == 'FiLM':
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                          f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss1, loss2, loss3, loss4, loss5, loss6 = criterion(pred, true)
                final_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                # final_loss = 0.5 * loss1 + 0.5 * loss2

                total_loss.append(final_loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader, _ = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        loss_log_path = os.path.join(self.args.checkpoints, setting, 'loss_log')
        loss_log_path = loss_log_path.replace('\\', '/')
        if not os.path.exists(loss_log_path):
            os.makedirs(loss_log_path)

        writer = SummaryWriter(log_dir=loss_log_path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                if batch is None:
                    continue
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -self.args.num_predict
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                        loss1, loss2, loss3, loss4, loss5, loss6 = criterion(outputs, batch_y)
                        final_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                        # final_loss = 0.5 * loss1 + 0.5 * loss2
                        train_loss.append(final_loss.item())
                else:

                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        if self.args.model == 'MoLE_RLinear' or self.args.model == 'MoLE_DLinear':
                            outputs = self.model(batch_x, batch_x_mark)
                        elif self.args.model == 'Autoformer':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            if self.args.model == 'FiLM':
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -self.args.num_predict
                    outputs_y = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:,
                              f_dim:].to(self.device)
                    loss1, loss2, loss3, loss4, loss5, loss6 = criterion(outputs_y, batch_y)
                    final_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                    train_loss.append(final_loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, final_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * \
                                ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    final_loss.backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    final_loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Validation', vali_loss, epoch + 1)
            writer.add_scalar('Loss/Test', test_loss, epoch + 1)
            writer.close()
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(
                    model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(
                    scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model

    def test(self, setting, test=1):

        test_data, test_loader, num_data = self._get_data(flag='test')
        self.num_data = num_data
        if test:
            print('loading model')
            path = os.path.join(
                r'./checkpoints/' + setting, 'checkpoint.pth')
            path = path.replace('\\', '/')

            self.model.load_state_dict(torch.load(path, map_location='cuda:0'))

        preds = []
        trues = []
        inputx = []
        folder_path = r'./outputs/' + setting + '/visual_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if batch is None:
                    continue
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            if self.args.model == 'MoLE_RLinear' or self.args.model == 'MoLE_DLinear':
                                outputs = self.model(batch_x, batch_x_mark)
                            elif self.args.model == 'Autoformer':
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        if self.args.model == 'MoLE_RLinear' or self.args.model == 'MoLE_DLinear':
                            outputs = self.model(batch_x, batch_x_mark)
                        elif self.args.model == 'Autoformer':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            if self.args.model == 'FiLM':
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()

                outputs_batch_size, outputs_sequence_length, outputs_num_features = outputs.shape
                batch_y_batch_size, batch_y_sequence_length, batch_y_num_features = batch_y.shape
                batch_x_batch_size, batch_x_sequence_length, batch_x_num_features = batch_x.shape
                outputs_2d = outputs.reshape(outputs_batch_size * outputs_sequence_length, outputs_num_features)
                batch_y_2d = batch_y.reshape(batch_y_batch_size * batch_y_sequence_length, batch_y_num_features)
                batch_x_2d = batch_x.reshape(batch_x_batch_size * batch_x_sequence_length, batch_x_num_features)
                outputs_2d = torch.FloatTensor(test_data.inverse_transform(outputs_2d)).to(self.device)
                batch_y_2d = torch.FloatTensor(test_data.inverse_transform(batch_y_2d))
                batch_x_2d = torch.FloatTensor(test_data.inverse_transform(batch_x_2d)).to(self.device)
                outputs = outputs_2d.reshape(outputs_batch_size, outputs_sequence_length, outputs_num_features)
                batch_y = batch_y_2d.reshape(batch_y_batch_size, batch_y_sequence_length, batch_y_num_features)
                batch_x = batch_x_2d.reshape(batch_x_batch_size, batch_x_sequence_length, batch_x_num_features)

                f_dim = -self.args.num_predict

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate(
                        (input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate(
                        (input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = self.padding_the_data(preds)
        trues = self.padding_the_data(trues)
        inputx = self.padding_the_data(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        inputx = inputx[:, :, f_dim:]

        self.save_preds(preds, flag='preds', setting=setting)
        self.save_preds(trues, flag='trues', setting=setting)
        self.save_preds(inputx, flag='inputx', setting=setting)
        self.AE(preds=preds, trues=trues, setting=setting)
        print('finish_save_metricx')

        folder_path = r'./outputs/' + setting + '/test_preresults/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, Max_error, Median_absolute_error, ev_scores, r2_scores, Adjust_R2_scores, STD = metric(
            pred=preds,
            true=trues,
            num_data=self.num_data,
            num_feature=self.args.num_predict,
            pred_len=self.args.pred_len
        )
        self.save_overall_evaluation(mae, mse, rmse, mape, mspe, rse, Max_error, Median_absolute_error, ev_scores,
                                     r2_scores, Adjust_R2_scores, STD, setting=setting)
        print('finish_save_overall_evaluation')
        self.every_channel_evaluation(preds=preds, trues=trues, num_data=self.num_data,
                                      num_feature=self.args.num_predict, pred_len=self.args.pred_len, setting=setting)
        print('finish_save_every_channel_evaluation')
        self.every_step_evaluation(preds=preds, trues=trues, num_data=self.num_data, num_feature=self.args.num_predict,
                                   pred_len=self.args.pred_len, setting=setting)
        print('finish_save_every_step_evaluation')

        print('mse:{}, rmse:{}, mae:{}, rse:{}, mape:{}, mspe:{},Max_error:{}, Median_absolute_error:{}, '
              'ev_scores:{}, r2_scores:{}, Adjust_R2_scores:{},STD:{}'.format(mse, rmse, mae, rse, mape, mspe,
                                                                              Max_error,
                                                                              Median_absolute_error, ev_scores,
                                                                              r2_scores,
                                                                              Adjust_R2_scores, STD))
        txt_path = os.path.join(r'./outputs/' + 'result.txt')
        f = open(txt_path, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, rmse:{}, mae:{}, rse:{}, mape:{}, mspe:{}, Max_error:{}, Median_absolute_error:{}, '
                'ev_scores:{}, r2_scores:{}, Adjust_R2_scores:{},STD:{}'.format(mse, rmse, mae, rse, mape, mspe,
                                                                                Max_error,
                                                                                Median_absolute_error, ev_scores,
                                                                                r2_scores,
                                                                                Adjust_R2_scores, STD))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = r'./checkpoints/real_prediction/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
