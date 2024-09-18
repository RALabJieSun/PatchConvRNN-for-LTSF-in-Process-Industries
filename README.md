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
- **Copyright Statement**： This     dataset and the associated work are provided by the State Key Laboratory     of Rolling and Automation (RAL), Northeastern University, Shenyang 110819,     Liaoning, PR China. The dataset is utilized exclusively for scientific     research purposes. If used, proper citation in the literature is required. Please     cite it as follows: ==**论文引用格式**==

## 2.5 Potential Application Areas for the Dataset

Engineers in the steel industry can use the dataset for precise control and optimization of production processes, while data scientists and AI experts can use it to develop and validate new time-series forecasting algorithms to drive process industry intelligence. In conjunction with the dataset, the performance of different models can be objectively evaluated, providing data to support further process improvements. At the same time, it provides the basis for creating more informative sensor layouts and improving equipment regulation strategies.



