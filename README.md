# Improving Large-scale Classification in Technology Management: Making Full Use of Label Information for Professional Technical Documents
## 使用依赖
```python
python ==3.6
torch==1.7.1
transformers==4.12.5 
spacy==3.4.4 
```
## 数据
```
USTPTO: Available at: http://mleg.cse.sc.edu/DeepPatent (accessed November 9,2022)
WIPO-alpha: Available at: https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset (accessed November 21, 2022)
```

## 相关说明
```--output：存放保存的模型
--data/process_data：存放训练数据
--data/data_analysis.py：训练数据统计分析脚本
--data/labels_data_proc.py：标签数据预处理
--data/data_proc.py：原始数据清洗过滤，构造训练数据集
--dataset.py：数据预处理成bert所需要的格式
--loss.py：损失函数实现
--models.py：模型网络结构代码
--train.py：主运行程序，包含训练、测试以及相关评价指标的计算
--bert-base-uncase:要预先下载好预训练的bert模型，放在和该项目同级下的bert-base-uncase文件夹下,需要的是vocab.txt、config.json、pytorch_model.bin
```
