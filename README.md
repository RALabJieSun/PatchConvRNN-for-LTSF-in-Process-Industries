# RAL_TCM
1. __Overview of the Dataset__

The dataset, named RAL\_TCM \(Tandem Cold\-rolling Mill\), is a large\-scale benchmark dataset designed for multivariate time series forecasting in the steel industry, particularly targeting complex process operations in process industries\. The RAL\_TCM dataset was collected from the tandem cold rolling production line of a steel company in Jiangsu Province, China, through Level\-2 process control systems and PDA systems\. It encompasses almost all critical process parameters involved in the tandem cold rolling production process\. 

1. __The Composition of the Dataset__

- __Data Source__: The tandem cold rolling production site of strip steel at a steel enterprise in Jiangsu Province, China\.
- __Collection System:__ Recorded through various sensors, instruments, and automatic control systems at the rolling production site, as well as data acquisition PDA systems\.
- __Data content__:
	- __\- Number of strip coils: 5__ 
	- __\- Time steps__： A total of 20,299 time steps with millisecond time granularity
	- __\- Number of __<a id="_Hlk174529752"></a>__process variables: __Each data point consists of 170 process variables
- __\- Target predicted variables \(8\)__：
	- 
		- Rolling forces for Stands 1 to 5:

RFC\_STD\#1

RFC\_STD\#2

RFC\_STD\#3

RFC\_STD\#4

RFC\_STD\#5

- 
	- 
		- Thickness of outlet strips for Stands 1, 4 and 5:

THK\_STD\#1, 

THK\_STD\#4, 

THK\_STD\#5

1. __Characteristics and split of the dataset__

- __High\-time resolution__: Millisecond data acquisition provides high temporal resolution data for high\-precision prediction tasks\.
- __Multivariate Temporal Sequencing__: Multivariate time\-series data with 170 process variables per time step captures multiple complex relationships in the production process and provides sufficient information for modelling\.
- __Large\-scale data__: The data volume of 20,299 time steps provides rich samples for model training and validation。
- __Split of the dataset__： The recommended split ratio of training, validation, and test sets is 7:1:2\.

__4\.	Dataset applications__

- __Application Fields__： The dataset can be used for research on topics such as machine learning, time\-series forecasting models, process optimization, production quality control in artificial intelligence, etc\.
- __Forecasting tasks__： The main task is to predict key process parameters \(e\.g\. rolling force and strip thickness\) that contribute to the productivity and product quality of the strip cold rolling process\.

__5\.	数据集获取与使用__

- <a id="_Hlk171778125"></a>__Github Open Source URL__： RAL\_TCM Dataset | Github\(https://github.com/TsingloongWang/RAL_TCM).
- __Copyright Statement__： This dataset and the associated work are provided by the State Key Laboratory of Rolling and Automation \(RAL\), Northeastern University, Shenyang 110819, Liaoning, PR China\. The dataset is utilized exclusively for scientific research purposes\. If used, proper citation in the literature is required\. Please cite it as follows: __论文引用格式__

__6\.	Potential application areas for the dataset__

Engineers in the steel industry can use the dataset for precise control and optimization of production processes, while data scientists and AI experts can use it to develop and validate new time\-series forecasting algorithms to drive process industry intelligence\. In conjunction with the dataset, the performance of different models can be objectively evaluated, providing data to support further process improvements\. At the same time, it provides the basis for creating more informative sensor layouts and improving equipment regulation strategies\.
