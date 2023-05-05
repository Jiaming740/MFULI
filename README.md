# MFULI
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
--tnse.py：标签聚类展示<br>
--loss.py：损失函数实现：FocalLoss+对比损失实现<br>
--models.py：模型网络结构代码<br>
--train.py：主运行程序，包含训练、测试以及相关评价指标的计算<br>
--bert-base-uncase:要预先下载好预训练的bert模型，放在和该项目同级下的bert-base-uncase文件夹下,需要的是vocab.txt、config.json、pytorch_model.bin
