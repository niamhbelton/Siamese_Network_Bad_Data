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
2. 'label' - binary values where one indicates that it is bad data
3. 'test' - binary values where one indicates that it is present in the test set.
4. 'ref_set' - binary values where one indicates that it is present in the training data for the Siamese Network.
5. 'iso_set' - binary values where one indicates that it is present in the training data for Isolation Forest.


An MRI was labelled as being bad data if it fell into any of the following categories;
1. Contained no anatomical information.
2. Contained no relevant anatomical information. For example, when training a model to detect ACL tears on sagittal data, an MRI from the axial or coronal plane is not informative.
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



## 4. Implementation of Isolation Forest for Identifying Bad Data

