# README

# 1. Usage

Read the environment requirements file 'requirements.txt', and download the missing support libraries for Python

Next, open the run.py configuration file and adjust the variants of codes as needed. The variants of codes below that need to be adjusted are:

```
"--target1": The process variable to be predicted can only be set to MS or S prediction modes in this code. Simply modify this section to change the variable that needs to be predicted.
"--is_training": training mode
```

The target predicted process variables (8) mentioned in the text are:

```
'TN\L2_AGC_S1TRFC_ACT': RFC_STD#1
'TN\L2_AGC_S2TRFC_ACT': RFC_STD#2
'TN\L2_AGC_S3TRFC_ACT': RFC_STD#3
'TN\L2_AGC_S4TRFC_ACT': RFC_STD#4
'TN\L2_AGC_S5TRFC_ACT': RFC_STD#5
'TN\L2_AGC_V2THK_ACT': THK_STD#1
'TN\L2_AGC_V5THK_ACT': THK_STD#4
'TN\L2_AGC_H5THK_ACT’: THK_STD#5
```

```
The key hyperparameters of the model and their meaning:
"--seq_len": The length of the review window
"--patch_len": enc_patch_len
"--pred_len": The predicted length
"--patch_pred_len":dec_patch_len
"--seq_cha": channel or dimension for sequence
"--enc_in": fus_out
Notes:
pred_len: the forecasting horizon; enc_patch_len: the length of each temporal patch extracted from the input sequence in the encoder; dec_patch_len: the length of each temporal patch in the decoder; fus_out: The output channels of the pointwise convolution; seq_cha: The number of variables or channels in the input sequence; seq_len: The number of time steps in the look-back window; conv_k: The kernel size used in depthwise convolution; dropout: The rate at which neurons are randomly set to zero during training to prevent overfitting; enc_in: The output dimensions or channels of Linear Projection before the RNN cell in the encoder.
```

Finally, if want to perform model prediction to verify the results presented in our paper, please modify the run.py setting on line-162 of the code. Change the setting to the filename of the model weight file for the desired results. For example:

```
setting = 'STD3_M100_50_50_50_mixed_std'
```

Before starting the training, please check lines 262 and 344 in the exp_main.py file. By default, mixed_std_loss is used. If you want to use mix_loss, please remove the relevant comments.

```
# final_loss = 0.5 * loss1 + 0.5 * loss2
```

## 1.1 Checkpoints

the checkpoints folder contains the pre-trained model weights for our PatchConvRNN, which achieved state-of-the-art forecasting results at different forecasting horizons as described in the paper (Tables 1 and 3).

{prediction variables}-{The predicted length}-{enc_patch_len}-{dec_patch_len}-{fus_out}-{loss}.

For example, the folder name STD1_M10_20_10_50_mixed_std can be interpreted as follows:  

STD1: RFC_STD#1

M10: the forecasting horizon=10

20: enc_patch_len=20

10: dec_patch_len=10

50: fus_out=50

mixed_std: mixed_std_loss

## 1.2 Outputs

The outputs folder contains visualization outputs and metrics files for model predictions. The initial .CSV file represents the result prediction data, while the "visual results" directory contains visual comparisons of sample prediction results.

test_preresults: Contains .npy files of the model's predicted values and the actual values.

visual results: Visualizations of the model's predicted values and actual values curves.

Absolute_Error.csv: The absolute error between the model's predicted values and the actual values.

every_channel_evaluation.csv: Evaluation metrics for each channel.

every_step_evaluation.csv: Evaluation metrics for each time step.

overall_evaluation.csv: Overall evaluation metrics.

test_inputs_X.csv: Model inputs.

test_preds_Y.csv: Model predictions.

test_trues_Y.csv: Actual values.



## 1.3 Training details and hyperparameter determinations

Our research uses the PyTorch deep learning framework for model construction, with experimentation conducted on the following computer hardware and software configurations: Linux operating system，2 × Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz，2 × NVIDIA GeForce RTX 4090, utilizing CUDA 9.2 as the GPU parallel computing platform. The detailed model training process: The batch size is 120, the learning rate is 0.001, the number of training epochs is 35, and the optimizer used is AdamW. The learning rate adjustment strategy involved maintaining a constant learning rate for the first three epochs, followed by a 10% decay for each subsequent epoch. the model hyperparameters for the best predictive performance are shown in **Table A1**.

<div align=center><strong>Table A1</strong> Hyperparameter determinations to achieve SOTA results for different forecasting horizons.</div>

| Forecasting tasks | pred_len | enc_patch_len | dec_patch_len | fus_out | seq_cha | seq_len | conv_k | dropout | enc_in |
| :---------------: | :------: | :-----------: | :-----------: | :-----: | :-----: | :-----: | :----: | :-----: | :----: |
|     RFC_STD#1     |    10    |      20       |      10       |   50    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |      25       |       5       |   50    |         |         |        |         |        |
|                   |    30    |      25       |       6       |   40    |         |         |        |         |        |
|                   |    40    |       4       |       5       |   50    |         |         |        |         |        |
|                   |    60    |       5       |       5       |   50    |         |         |        |         |        |
|                   |    80    |       4       |      10       |   50    |         |         |        |         |        |
|                   |   100    |      25       |      10       |   50    |         |         |        |         |        |
|     RFC_STD#2     |    10    |      10       |      10       |   40    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |      10       |       2       |   50    |         |         |        |         |        |
|                   |    30    |       4       |      15       |   20    |         |         |        |         |        |
|                   |    40    |       4       |       5       |   10    |         |         |        |         |        |
|                   |    60    |      25       |       2       |   30    |         |         |        |         |        |
|                   |    80    |      50       |       5       |   20    |         |         |        |         |        |
|                   |   100    |      25       |       5       |   20    |         |         |        |         |        |
|     RFC_STD#3     |    10    |      20       |       5       |   50    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |      50       |       5       |   30    |         |         |        |         |        |
|                   |    30    |      20       |       2       |   50    |         |         |        |         |        |
|                   |    40    |      50       |       5       |   50    |         |         |        |         |        |
|                   |    60    |      20       |      30       |   50    |         |         |        |         |        |
|                   |    80    |      25       |      40       |   50    |         |         |        |         |        |
|                   |   100    |      50       |      50       |   50    |         |         |        |         |        |
|     RFC_STD#4     |    10    |       5       |       5       |   50    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |      100      |       5       |   50    |         |         |        |         |        |
|                   |    30    |      10       |       5       |   50    |         |         |        |         |        |
|                   |    40    |      100      |       5       |   50    |         |         |        |         |        |
|                   |    60    |      25       |      10       |   50    |         |         |        |         |        |
|                   |    80    |       4       |       5       |   50    |         |         |        |         |        |
|                   |   100    |      100      |       5       |   10    |         |         |        |         |        |
|     RFC_STD#5     |    10    |      20       |      10       |   30    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |       5       |       5       |   50    |         |         |        |         |        |
|                   |    30    |      20       |       5       |   50    |         |         |        |         |        |
|                   |    40    |      100      |       5       |   50    |         |         |        |         |        |
|                   |    60    |      10       |      60       |   50    |         |         |        |         |        |
|                   |    80    |       2       |      20       |   50    |         |         |        |         |        |
|                   |   100    |      10       |      50       |   130   |         |         |        |         |        |
|     THK_STD#1     |    10    |      50       |       5       |   50    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |       5       |      10       |   50    |         |         |        |         |        |
|                   |    30    |      25       |       5       |   50    |         |         |        |         |        |
|                   |    40    |      10       |       8       |   50    |         |         |        |         |        |
|                   |    60    |       5       |      40       |   90    |         |         |        |         |        |
|                   |   100    |      100      |      20       |   50    |         |         |        |         |        |
|     THK_STD#4     |    10    |      50       |       5       |   40    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |      20       |      10       |   20    |         |         |        |         |        |
|                   |    30    |      25       |       5       |   50    |         |         |        |         |        |
|                   |    40    |      20       |      10       |   50    |         |         |        |         |        |
|                   |    60    |      25       |      15       |   50    |         |         |        |         |        |
|                   |    80    |      25       |       5       |   50    |         |         |        |         |        |
|                   |   100    |      25       |      20       |   50    |         |         |        |         |        |
|     THK_STD#5     |    10    |      50       |       5       |   50    |   170   |   100   |  3×3   |   0.5   |  256   |
|                   |    20    |      20       |      10       |   30    |         |         |        |         |        |
|                   |    30    |      25       |      155      |   50    |         |         |        |         |        |
|                   |    40    |      10       |       5       |   50    |         |         |        |         |        |
|                   |    60    |      50       |       5       |   50    |         |         |        |         |        |
|                   |    80    |      50       |      10       |   50    |         |         |        |         |        |
|                   |   100    |      25       |      10       |   50    |         |         |        |         |        |

**Note.** pred_len: the forecasting horizon; enc_patch_len: the length of each temporal patch extracted from the input sequence in encoder; dec_patch_len: the length of each temporal patch in decoder; fus_out: The output channels of the pointwise convolution; seq_cha: The number of variables or channels in the input sequence; seq_len: The number of time steps in the look-back window; conv_k: The kernel size used in depthwise convolution; dropout: The rate at which neurons are randomly set to zero during training to prevent overfitting; enc_in: The output dimensions or channels of Linear Projection before the RNN cell in encoder.

<div align=center><strong>Table A2 Effects of temporal patch length in the encoder on prediction results.</strong> The forecasting horizon T is 100 and the best results are highlighted in <strong>bold</strong>.</div>

| enc_patch_len | RFC_STD#1 (RMSE) | RFC_STD#1 (MAPE) | RFC_STD#2 (RMSE) | RFC_STD#2 (MAPE) | RFC_STD#3 (RMSE) | RFC_STD#3 (MAPE) | RFC_STD#4 (RMSE) | RFC_STD#4 (MAPE) | RFC_STD#5 (RMSE) | RFC_STD#5 (MAPE) | THK_STD#1 (RMSE) | THK_STD#1 (MAPE) | THK_STD#4 (RMSE) | THK_STD#4 (MAPE) | THK_STD#5 (RMSE) | THK_STD#5 (MAPE) |
| ------------- | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| 2             |     159.027      |      1.231       |      52.869      |      0.396       |      52.837      |      0.472       |      48.756      |      0.479       |    **32.553**    |    **0.310**     |      4.027       |      1.632       |      1.201       |      1.594       |      1.120       |      1.782       |
| 4             |     158.643      |      1.230       |      51.528      |      0.384       |      46.725      |      0.385       |      48.648      |      0.479       |      33.439      |      0.340       |      4.027       |      1.632       |      1.159       |      1.538       |      1.107       |      1.762       |
| 5             |     153.044      |      1.192       |      52.025      |      0.391       |      47.635      |      0.386       |      44.994      |      0.466       |      34.991      |      0.340       |      3.876       |      1.566       |      1.158       |      1.533       |      1.062       |      1.692       |
| 10            |     161.596      |      1.261       |      51.866      |      0.386       |      46.206      |      0.410       |      44.613      |      0.462       |      42.356      |      0.434       |      4.085       |      1.663       |      1.183       |      1.571       |      1.105       |      1.761       |
| 20            |     175.788      |      1.376       |      53.041      |      0.398       |      47.271      |      0.366       |      44.647      |      0.462       |      41.311      |      0.424       |      4.183       |      1.706       |      1.158       |      1.535       |      1.116       |      1.779       |
| 25            |   **150.669**    |    **1.172**     |    **50.492**    |    **0.381**     |      49.511      |      0.418       |      44.240      |      0.458       |      40.270      |      0.412       |      4.006       |      1.633       |    **1.126**     |    **1.491**     |    **1.045**     |    **1.663**     |
| 50            |     159.961      |      1.245       |      52.276      |      0.394       |    **45.850**    |    **0.380**     |      43.964      |      0.455       |      35.860      |      0.351       |      4.076       |      1.667       |      1.183       |      1.568       |      1.065       |      1.694       |
| 100           |     182.514      |      1.409       |      53.720      |      0.405       |      49.889      |      0.421       |    **43.669**    |    **0.452**     |      35.860      |      0.317       |    **3.821**     |    **1.538**     |      1.175       |      1.557       |      1.065       |      1.695       |

<div align=center><strong>Table A3 Effects of temporal patch length in the decoder on prediction results.</strong> The forecasting horizon T is 100 and the best results are highlighted in <strong>bold</strong>.</div>

| dec_patch_len | RFC_STD#1 (RMSE) | RFC_STD#1 (MAPE) | RFC_STD#2 (RMSE) | RFC_STD#2 (MAPE) | RFC_STD#3 (RMSE) | RFC_STD#3 (MAPE) | RFC_STD#4 (RMSE) | RFC_STD#4 (MAPE) | RFC_STD#5 (RMSE) | RFC_STD#5 (MAPE) | THK_STD#1 (RMSE) | THK_STD#1 (MAPE) | THK_STD#4 (RMSE) | THK_STD#4 (MAPE) | THK_STD#5 (RMSE) | THK_STD#5 (MAPE) |
| :-----------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|       2       |     157.926      |      1.252       |      51.938      |      0.390       |      49.755      |      0.443       |      45.947      |      0.476       |      41.994      |      0.426       |      3.919       |      1.590       |      1.191       |      1.578       |      1.092       |      1.741       |
|       4       |     151.308      |      1.188       |      51.580      |      0.386       |      43.550      |      0.384       |      45.958      |      0.476       |      42.322      |      0.431       |      4.004       |      1.625       |      1.136       |      1.505       |      1.081       |      1.723       |
|       5       |     150.669      |      1.172       |    **50.492**    |    **0.381**     |      45.850      |      0.380       |    **43.669**    |    **0.452**     |      32.553      |      0.310       |      3.821       |      1.538       |      1.126       |      1.491       |      1.045       |      1.663       |
|      10       |   **149.971**    |    **1.164**     |      51.521      |      0.386       |      49.957      |      0.413       |      45.391      |      0.470       |    **32.040**    |    **0.310**     |      3.797       |      1.535       |      1.117       |      1.480       |    **1.035**     |    **1.648**     |
|      20       |     153.617      |      1.177       |      50.950      |      0.381       |      45.781      |      0.380       |      47.677      |      0.492       |      40.511      |      0.413       |    **3.794**     |    **1.532**     |    **1.105**     |    **1.461**     |      1.059       |      1.680       |
|      25       |     162.937      |      1.241       |      51.885      |      0.389       |      46.322      |      0.377       |      45.333      |      0.470       |      40.193      |      0.410       |      3.904       |      1.562       |      1.108       |      1.463       |      1.065       |      1.696       |
|      50       |     161.299      |      1.260       |      51.534      |      0.386       |    **40.143**    |    **0.353**     |      47.358      |      0.490       |      40.065      |      0.409       |      4.005       |      1.634       |      1.108       |      1.465       |      1.043       |      1.658       |
|      100      |     160.624      |      1.242       |      51.339      |      0.387       |      44.551      |      0.394       |      44.394      |      0.460       |      38.301      |      0.382       |      3.923       |      1.591       |      1.110       |      1.469       |      1.040       |      1.652       |

<div align=center><strong>Table A4 Effects of channel-fusion strategy of multivariate on the prediction results.</strong> The forecasting horizon T is 100, and the number of variables or channels in the input time series is 170. The best results are highlighted in <strong>bold</strong>.</div>

| fus_out | RFC_STD#1 (RMSE) | RFC_STD#1 (MAPE) | RFC_STD#2 (RMSE) | RFC_STD#2 (MAPE) | RFC_STD#3 (RMSE) | RFC_STD#3 (MAPE) | RFC_STD#4 (RMSE) | RFC_STD#4 (MAPE) | RFC_STD#5 (RMSE) | RFC_STD#5 (MAPE) | THK_STD#1 (RMSE) | THK_STD#1 (MAPE) | THK_STD#4 (RMSE) | THK_STD#4 (MAPE) | THK_STD#5 (RMSE) | THK_STD#5 (MAPE) |
| :-----: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|   10    |     150.260      |      1.193       |      50.532      |      0.381       |      45.768      |      0.374       |    **42.734**    |    **0.443**     |      47.162      |      0.479       |      3.893       |      1.589       |      1.134       |      1.503       |      1.076       |      1.716       |
|   20    |     155.040      |      1.206       |    **50.343**    |    **0.376**     |      44.989      |      0.361       |      43.502      |      0.451       |      46.096      |      0.470       |      4.103       |      1.684       |      1.115       |      1.477       |      1.065       |      1.697       |
|   30    |     151.476      |      1.188       |      51.945      |      0.390       |      47.769      |      0.391       |      48.968      |      0.506       |      41.573      |      0.424       |      3.866       |      1.557       |      1.116       |      1.476       |      1.048       |      1.668       |
|   40    |     160.363      |      1.250       |      52.183      |      0.392       |      44.719      |      0.365       |      44.704      |      0.463       |      44.015      |      0.449       |      3.881       |      1.573       |      1.113       |      1.474       |      1.063       |      1.694       |
|   50    |   **149.971**    |    **1.164**     |      50.492      |      0.381       |    **40.143**    |    **0.353**     |      43.669      |      0.452       |      32.040      |      0.310       |    **3.794**     |    **1.532**     |    **1.105**     |    **1.461**     |    **1.035**     |    **1.648**     |
|   70    |     171.479      |      1.309       |      52.139      |      0.391       |      42.172      |      0.353       |      45.660      |      0.472       |      33.855      |      0.323       |      3.868       |      3.868       |      1.113       |      1.469       |      1.099       |      1.750       |
|   90    |     152.871      |      1.180       |      53.285      |      0.403       |      41.010      |      0.360       |      43.874      |      0.454       |    **31.734**    |    **0.321**     |      4.182       |      1.705       |      1.120       |      1.483       |      1.057       |      1.683       |
|   100   |     163.027      |      1.252       |      53.142      |      0.396       |      42.073      |      0.371       |      43.932      |      0.454       |      39.714      |      0.407       |      3.968       |      1.612       |      1.126       |      1.490       |      1.054       |      1.678       |
|   130   |     160.070      |      1.223       |      52.750      |      0.385       |      41.449      |      0.366       |      45.396      |      0.470       |      39.658      |      0.406       |      3.985       |      1.625       |      1.224       |      1.626       |      1.043       |      1.661       |
|   170   |     166.832      |      1.300       |      51.250      |      0.397       |      41.561      |      0.368       |      44.237      |      0.458       |      33.798      |      0.322       |      3.985       |      1.616       |      1.223       |      1.620       |      1.059       |      1.684       |

## 1.4 Mixed-STD Loss Result

To mitigate the inductive bias introduced by traditional loss functions, we propose the **Mixed-STD loss** function. This novel loss function integrates both individual time-step errors and the statistical properties of the entire forecasting horizon. The **Mixed-STD loss** substantially reduces discrepancies between the predicted and actual value distributions, thereby capturing significant fluctuations in process variables more accurately. Consequently, the predicted sequences more closely follow the actual data trends. 

<img src='fig/Mixed-STD loss result1.jpg' width='1000'>





# 2. RAL_TCM Dataset

## 2.1 Overview of the Dataset

The dataset, named RAL_TCM (Tandem Cold-rolling Mill), is a large-scale benchmark dataset designed for multivariate time series forecasting in the steel industry, particularly targeting complex process operations in process industries. The RAL_TCM dataset was collected from the tandem cold rolling production line of a steel company in Jiangsu Province, China, through Level-2 process control systems and PDA systems. It encompasses almost all critical process parameters involved in the tandem cold rolling production process.

## 2.2 The Composition of the Dataset

* **Data Source**: The tandem cold rolling production site of strip steel at a steel enterprise in Jiangsu Province, China.

* **Collection System:** Recorded through various sensors, instruments, and automatic control systems at the rolling production site, as well as data acquisition PDA systems.

* **Data content**:
  - **Number of strip coils: 5** 
  -  **Time steps** :A total of 20,299 time steps with millisecond time granularity
  -  Use a **blank** line to separate each coils.

- **Number of process variables:** Each data point consists of 170 process variables

- **Target predicted variables (8):**

  - Rolling forces for Stands 1 to 5:

    RFC_STD#1

    RFC_STD#2

    RFC_STD#3

    RFC_STD#4

    RFC_STD#5

  - Thicknesses of outlet strips for Stands 1, 4 and 5:

    THK_STD#1

    THK_STD#4

    THK_STD#5

## 2.3 **Characteristics and Split of the Dataset** 

* **High-time resolution:** Millisecond data acquisition provides high temporal resolution data for high-precision prediction tasks.

+ **Multivariate Temporal Sequencing**: Multivariate time-series data with 170 process variables per time step captures multiple complex relationships in the production process and provides sufficient information for modelling.

+ **Large-scale data**: The data volume of 20,299 time steps provides rich samples for model training and validation。

+ **Split of the dataset**： The recommended split ratio of training, validation, and test sets is 7: 1: 2.

## 2.4 **Dataset Acquisition and Use**

- **Github** **Open Source URL**： RAL_TCM Dataset | RALabJieSun/PatchConvRNN-for-LTSF-in-Process-Industries (github.com)](https://github.com/RALabJieSun/PatchConvRNN-for-LTSF-in-Process-Industries).
- **Copyright Statement**： This     dataset and the associated work are provided by the State Key Laboratory     of Rolling and Automation (RAL), Northeastern University, Shenyang 110819,     Liaoning, PR China. The dataset is utilized exclusively for scientific     research purposes. If used, proper citation in the literature is required. Please     cite it as follows: ==**LATER**==

## 2.5 Potential Application Areas for the Dataset

Engineers in the steel industry can use the dataset for precise control and optimization of production processes, while data scientists and AI experts can use it to develop and validate new time-series forecasting algorithms to drive process industry intelligence. In conjunction with the dataset, the performance of different models can be objectively evaluated, providing data to support further process improvements. At the same time, it provides the basis for creating more informative sensor layouts and improving equipment regulation strategies.



