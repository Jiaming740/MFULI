# Improving Large-scale Classification in Technology Management: Making Full Use of Label Information for Professional Technical Documents
This repository implements MFULI model for hierarchical multi-label professional technical document classification. This work has been published as the Research Article "*Improving Large-scale Classification in Technology Management: Making Full Use of Label Information for Professional Technical Documents*" by the journal of *IEEE Transactions on Engineering Management*. You can download the code for your own research. For any further inquiries, please feel free to contact Dr. Jiaming Ding (email: *djm@mail.hfut.edu.cn*) .


## Directory Structure
```
├── data/                        # Directory for data-related scripts and files
│   ├── process_data/            # Directory containing the training data
│   ├── CPCTitleList202208/      # Directory containing CPC title lists
│   ├── data_analysis.py         # Script for training data statistical analysis
│   ├── labels_data_proc.py      # Script for label data preprocessing
│   └── data_proc.py             # Script for cleaning raw data and constructing the training dataset
├── dataset.py                   # Script to convert data into the format required by BERT
├── losses.py                      # Implementation of custom loss functions
├── models.py                    # Code for model architecture and network structure
├── train.py                     # Main script for running training, testing, and calculating evaluation metrics
```

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

The training data needs to be processed into the following format and stored as TSV files:
patent_id,title,abstract,labels
9009180,System and method for providing extending searches,"The present invention generally relates to computer and web-based contact searches. Specifically, this invention relates to systems and methods for extending contact searches to include contacts beyond those of the user initiating the search. Embodiments of the present invention allow users to search for indirect contacts beyond their direct contacts by providing the user results that include the contacts of their contacts and so on to a specified depth level and restricted by security implementations selectable by the indirect contacts.","['Y10S', 'H04L', 'G06F']"
```

## Installation
```
To clone this repository, run the following command in your terminal:
git clone https://github.com/Jiaming740/MFULI.git
```

## Citation
```
If you think the code is useful in your research, please kindly consider to refer:

@article{ding2024improving,
  title={Improving Large-Scale Classification in Technology Management: Making Full Use of Label Information for Professional Technical Documents},
  author={Ding, Jiaming and Wang, Anning and Huang, Kenneth Guang-Lih and Zhang, Qiang and Yang, Shanlin},
  journal={IEEE Transactions on Engineering Management},
  year={2024},
  publisher={IEEE}
}
```
