## 使用依赖
```python
python ==3.6
torch==1.7.1
transformers==4.12.5 
spacy==3.4.4 
```
## 相关说明
--output：存放保存的模型<br>
--data/process_data：存放训练数据<br>
--data/data_analysis.py：训练数据统计分析脚本<br>
--data/labels_data_proc.py：标签数据预处理<br>
--data/data_proc.py：原始数据清洗过滤，构造训练数据集<br>
--dataset.py：数据预处理成bert所需要的格式<br>
--tnse.py：标签聚类展示
--loss.py：损失函数实现：FocalLoss+对比损失实现<br>
--models.py：模型网络结构代码
--dep_parser.py：依存句法分析依存关系矩阵实现
--GCN.py：图神经网络实现代码
--train.py：主运行程序，包含训练、测试以及相关评价指标的计算<br>
--bert-base-uncase:要预先下载好预训练的bert模型，放在和该项目同级下的bert-base-uncase文件夹下,需要的是vocab.txt、config.json、pytorch_model.bin

# 运行结果：
【bert】      
ham_loss：0.012844 micro_precision：0.7651 micro_recall：0.6938 micro_f1：0.7277
accuracy：0.4921 macro_precision：0.7207 macro_recall：0.5956 macro_f1：0.6310

【bert contrast】
【test】 ham_loss：0.013323 micro_precision：0.7337 micro_recall：0.7242 micro_f1：0.7289
【test】accuracy：0.4860 macro_precision：0.6777 macro_recall：0.6306 macro_f1：0.6459

【bert contrast+LMB】
【test】 ham_loss：0.013273 micro_precision：0.7367 micro_recall：0.7212 micro_f1：0.7289
【test】accuracy：0.4887 macro_precision：0.6791 macro_recall：0.6400 macro_f1：0.6518

【bert contrast+LMB+FL】
【test】 ham_loss：0.014738 micro_precision：0.6848 micro_recall：0.7490 micro_f1：0.7155
【test】accuracy：0.4547 macro_precision：0.6376 macro_recall：0.6636 macro_f1：0.6456