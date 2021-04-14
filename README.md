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


## 3. Implementation of the Siamese Network for Identifying Bad Data



## 4. Implementation of Isolation Forest for Identifying Bad Data

