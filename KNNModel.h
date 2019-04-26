#ifndef KNNMODEL_H_INCLUDED
#define KNNMODEL_H_INCLUDED

#include <string>
#include <vector>
using namespace std;

class KNNModel {

private:
    size_t pixelCount;
    vector<vector<double>> features;
    vector<int> labels;
    vector<size_t> keys;

public:
    KNNModel(size_t _pixelCount = 784): pixelCount(_pixelCount) {}
    KNNModel(vector<vector<double>> _features, vector<int> _labels, vector<size_t> _keys, size_t _pixelCount = 784): features(_features), labels(_labels), keys(_keys), pixelCount(_pixelCount) {}
    void train(string filename = "train.csv");
    vector<int> predict(string filename = "test.csv", size_t K = 11);
    double crossValidate(size_t fold = 10, size_t K = 11);
    size_t getPixelCount() {return pixelCount;}
    vector<vector<double>> getFeatures() {return features;}
    vector<int> getLabels() {return labels;}
    vector<size_t> getKeys() {return keys;}
};

#endif // KNNMODEL_H_INCLUDED
