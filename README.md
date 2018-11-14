# Purr -- A Multi-label Text Classification Tool
🐱🐱🐱
## Table of contents
[Brief Introduction](#Brief-Introduction)  
[Dependencies](#Dependencies)  
[Commands](#Commands)  
[Data Folder Structure](#Data-Folder-Structure)  
[Acknowledgement](#Acknowledgement)

## Brief Introduction 
 
**Purr is a tool for training multi-label text classification neural networks.** 

You can do data preprocessing, training, testing and inferencing using this tool. There is one model with this tool now, which is a CNN-Attention based model. It computes the probabilities of the input text belonging to each label and outputs binary numbers as results. For example, if there are in total 5 possible labels, [1, 0, 0, 1, 0] means the text belongs to label 1 and 4.

To use this tool, please follow these steps (commands in the next section):  
1. Put raw data in the `data` folder and the format should be like `toy_data`.
2. Create folder to save model.
3. Define some constants like data path and model configs in the `constants.py` file.
4. Run the `preprocess.py` script.
5. Modify the training script and run it.
6. After obtaining the model, define some constants in the `inference.py` script and run it for inferencing.

## Dependencies
This project is based on `python>=3.6`. The dependent package for this project is listed as below:
```
- pandas==0.20.3
- torch==0.3.1
- scipy==0.19.1
- numpy==1.14.2
- gensim==2.3.0
- scikit_learn==0.20.0
- tqdm==4.28.1
```

## Commands
- create necessary folders
```
mkdir saved_models
```
- preprocess data
```
python preprocess.py data/toy_data
```
- train
```
./train.sh
```

## Data Folder Structure
```
├── data
│   ├── toy_data
│   ├── data.csv
│   ├── label_list.csv
│   ├── processed.embed
│   ├── processed.w2v
│   ├── test.csv
│   ├── train.csv
│   └── vocab.csv
```
## Acknowledgement
- Inspired by [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)