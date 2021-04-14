# Semi-Supervised Siamese Network for Identifying Bad Data in Medical Imaging Datasets

This repository is under construction.

## Contents
1. Data.
2. Splitting the data and labelling process.
3. Implementation of the Siamese Network for identifying bad data.
4. Implementation of Isolation Forest for identifying bad data.



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

## 3. Implementation of the Siamese Network for Identifying Bad Data
```
df
```


## 4. Implementation of Isolation Forest for Identifying Bad Data
The implemetation of Isolation Forest can be found in isolation_forest.ipynb. The features in the model were pixel histograms. Six bins were used in this analysis.

