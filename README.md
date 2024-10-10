# Improving Large-scale Classification in Technology Management: Making Full Use of Label Information for Professional Technical Documents
This repository implements MFULI model for hierarchical multi-label professional technical document classification. This work has been accepted as the Research Article "Improving Large-scale Classification in Technology Management: Making Full Use of Label Information for Professional Technical Documents" in *IEEE Transactions on Engineering Management*.

## Dependencies
```
Python == 3.6 – The project is compatible with Python 3.6.
torch == 1.7.1 – PyTorch library for building and training deep learning models.
transformers == 4.12.5 – Hugging Face's library for pre-trained transformer models like BERT.
spacy == 3.4.4 – NLP library for advanced tokenization and linguistic features.
```
## Data
```
USPTO: Available at: http://mleg.cse.sc.edu/DeepPatent (accessed November 9,2022)
WIPO-alpha: Available at: https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset (accessed November 21, 2022)
```

## Directory Structure
```
├── data/                        # Directory for data-related scripts and files
│   ├── process_data/            # Directory containing the training data
│   ├── CPCTitleList202208/      # Directory containing CPC title lists
│   ├── data_analysis.py         # Script for training data statistical analysis
│   ├── labels_data_proc.py      # Script for label data preprocessing
│   └── data_proc.py             # Script for cleaning raw data and constructing the training dataset
├── dataset.py                   # Script to convert data into the format required by BERT
├── loss.py                      # Implementation of custom loss functions
├── models.py                    # Code for model architecture and network structure
├── train.py                     # Main script for running training, testing, and calculating evaluation metrics
```

## Installation
```
To clone this repository, run the following command in your terminal:
git clone https://github.com/Jiaming740/MFULI.git
```

