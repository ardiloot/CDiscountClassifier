[![Build status](https://ci.appveyor.com/api/projects/status/8uhf7fca7d00vso3/branch/master?svg=true)](https://ci.appveyor.com/project/ardiloot/cdiscountclassifier/branch/master)
[![Build Status](https://travis-ci.org/ardiloot/CDiscountClassifier.svg?branch=master)](https://travis-ci.org/ardiloot/CDiscountClassifier)

# CDiscountClassifier

This code was used to train Xception model for Kaggle competition [Cdiscount’s Image Classification Challenge](https://www.kaggle.com/c/cdiscount-image-classification-challenge). In principle, the competition was about standard image classification, however following points made it difficult:

* Large number of categories (5720)
* Big training data (58.2 GB, 7 million products each having up to 4 images)
* A lot of difficult categories (different styles of books and CDs)
* Variable number of images (1-4) for each product

## Requirements

* keras
* tensorflow
* pandas
* numpy
* scipy
* pymongo
* scikit-image
* pillow
* cython
* h5py

## Results

The final results of the competition are available [here](https://www.kaggle.com/c/cdiscount-image-classification-challenge/leaderboard). The Xception model (team name "Ardi Loot") trained with this code got 64th position (bronze) out of 627 (accuracy 0.72582).

 
 