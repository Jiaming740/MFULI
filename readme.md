## Dependencies
```python
python ==3.6
torch==1.7.1
transformers==4.12.5 
spacy==3.4.4 
```
## Project Structure
```
--output: Directory to store the saved models.
--data/process_data: Directory for storing the training data.
--data/data_analysis.py: Script for performing statistical analysis on the training data.
--data/labels_data_proc.py: Script for label data preprocessing.
--data/data_proc.py: Script for cleaning raw data and constructing the training dataset.
--dataset.py: Script for converting data into the format required by BERT.
--losses.py: Implementation of loss functions, including FocalLoss and contrastive loss.
--models.py: Code for the model architecture and network structure.
--train.py: Main script for running training, testing, and calculating evaluation metrics.
--bert-base-uncase: You need to pre-download the BERT model (bert-base-uncase) and place it in a directory at the same level as this project. This folder should contain the files vocab.txt, config.json, and pytorch_model.bin.
```
