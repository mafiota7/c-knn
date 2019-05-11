There have several variables you may want to change before compiling main.cpp and running it:

trainFilename - this is the name of the training data set (default - "train.csv")
unlabeledFilename - this is the name of the test data set (default - "test.csv")
outputFilename - this is the name of the file in which the result will be saved (default - "result.txt")
K - this is the number of neighbors (default - 3)
Nfold - when this is >= 2 then cross-validation of the training set will be performed (default - 0)
components - this is how many of the features to use. when it is = 0 then all features will be used (default - 0)