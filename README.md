# Purr -- A Multi-label Text Classification Tool
🐱🐱🐱
## Table of contents
[Brief Introduction](#Brief-Introduction)  
[Dependencies](#Dependencies)  
[Commands](#Commands)  
[Data Folder Structure](#Data-Folder-Structure)  
[Acknowledgement](#Acknowledgement)

## Brief Introduction 
 
Purr is a tool for training multi-label text classification neural networks.  
raw data should be put in the `data` folder and the format should be like `toy_data`.

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