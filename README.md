# Semi-Supervised Siamese Network for Identifying Bad Data in Medical Imaging Datasets

This is the repository for '[Semi-Supervised Siamese Network for Identifying Bad Data in Medical Imaging Datasets](https://arxiv.org/pdf/2108.07130.pdf)'.

## Citation 
Belton, N., Lawlor, A., Curran, K., "Semi-Supervised Siamese Network for Identifying Bad Data in Medical Imaging Datasets" Medical Imaging with Deep Learning (MIDL), 2021. https://arxiv.org/pdf/2108.07130.pdf

## Abstract
Noisy data present in medical imaging datasets can often aid the development of robust models that are equipped to handle real-world data. However, if the bad data contains insufficient anatomical information, it can have a severe negative effect on the model's performance. We propose a novel methodology using a semi-supervised Siamese network to identify bad data. This method requires only a small pool of 'reference' medical images to be reviewed by a non-expert human to ensure the major anatomical structures are present in the Field of View. The model trains on this reference set and identifies bad data by using the Siamese network to compute the distance between the reference set and all other medical images in the dataset. This methodology achieves an Area Under the Curve (AUC) of 0.989 for identifying bad data. 


![GitHub Logo](/Images/figure_paper.png)



## Contents
1. Data.
2. Splitting the data and labelling process.
3. Description of a basic Data Loader that loads clean data only.
4. Models

    (A) Implementation of the Siamese Network for identifying bad data.
    
    (B) Implementation of Isolation Forest for identifying bad data.



## 1. Data
The MRNet dataset is available at https://stanfordmlgroup.github.io/competitions/mrnet/. This dataset consists of 1,250 knee MRIs acquired from three planes; axial, coronal and sagittal. It has image level labels of ACL tear, meniscus tear and/or abnormal. This project uses only the sagittal plane data.   


## 2. Splitting the Data and Labelling Process
The splitting of the dataset is outlined in metadata.csv. The columns of the csv are as follow;
1. 'id' - MRI ID
2. 'label' - binary values where one indicates that it is bad data.
3. 'test' - binary values where one indicates that it is present in the test set. There are 739 MRIs in this set.
4. 'ref_set' - binary values where one indicates that it is present in the training data for the Siamese Network. There are 20 MRIs in the this set.
5. 'iso_set' - binary values where one indicates that it is present in the training data for Isolation Forest. There are 500 MRIs in this set. 
6. 'mrnet_split' - binary values where one indicates that this case is in the official MRNet validation set. This feature is needed to locate the data i.e. if the data is from the official validation set, it will be stored in the valid folder, otherwise it is stored in the train folder.


An MRI was labelled as being bad data if it fell into any of the following categories;
1. Contained no anatomical information.
2. Contained no relevant anatomical information. For example, when training a model to detect ACL tears on sagittal data, an MRI from the axial or coronal plane that has been included in the sagittal plane data by mistake will harm the model's performance. 
3. Most of the important anatomical information is out of view.

The MRI cases that were considered to be bad data are highlighted in the table below.
| MRI ID | Category | Further description |
| :---: | :---: | :---: |
| 0003 | 1 | - |
| 0275 | 2 | Data Acquired from Coronal Plane |
| 0544 | 3 | - |
| 0582 | 1 | - |
| 0864 | 2 | Data Acquired from Coronal Plane |
| 1159 | 2 | Data Acquired from Coronal Plane |
| 1230 | 2 | Data Acquired from Axial Plane |


## 3. Description of a basic Data Loader that loads clean data only.
The directory Clean_Dataloader is a basic data loader written in Pytorch that loads the clean data only. The init function removes the bad data from the dataset. 

## 4. Models
### (A) Implementation of the Siamese Network for Identifying Bad Data
The code to implement the Siamese network is available in the 'Siamese' folder. The sensitivity and specificity was calculated based on the largest Euclidean Distance between reference images in the final epoch. This value was 0.9166.


### Training
The 'train.py', 'dataloader.py' and 'model.py' scripts are used to train the model. These scripts assume the 'metadata.csv' is in your current working directory. The training script takes the following arguments;
```
parser.add_argument('-m', '--model_name', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--data_path',  required=True)
parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
```
'-m' is the name to call the trained model.

'--epochs' is to specify the number of epochs.

'--data_path' is the path to where the MRNet data is stored.

'-i' is the indexes of the MRI cases that will be used as the reference images. They should be input as a string separated by a comma and whitespace. For example, '45, 904, 999, 456'. This argument is not required. The script defaults to 20 reference images.

This script will create an 'outputs' directory and it will output the trained model into the outputs directory. An example of how to run the script is shown below;
```
python train.py -m model1 --epochs 6 --data_path '/docs/siamese/MRNet/data/'
```

### Testing
The 'evaluate.py' script will test the model on the test set. The evaluate script takes the following arguments;
```
parser.add_argument('-m', '--model_name', type=str, required=True)
parser.add_argument('-o', '--output_name', type=str, required=True)
parser.add_argument('--data_path',  required=True)
parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
```
'-m' is the name of the model that is being tested. This is in the 'outputs' directory.

'-o' is the name of the file to write out the Mean Euclidean Distance of each test case. This will be written out the the 'outputs' directory.

'--data_path' is the path to where the MRNet data is stored.

'-i' is the indexes of the MRI cases that will be used as the reference images. They should be input as a string separated by a comma and whitespace. For example, '45, 904, 999, 456'. This argument is not required. The script defaults to 20 reference images.

An example of how to run the script to test the model is shown below;
```
python evaluate.py -m model1 -o results.csv --data_path '/docs/siamese/MRNet/data/'
```


### (B) Implementation of Isolation Forest for Identifying Bad Data
The implemetation of Isolation Forest can be found in isolation_forest.ipynb. The features in the model were pixel histograms. Six bins were used in this analysis.

