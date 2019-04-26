#include "KNNModel.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
using namespace std;

const int epsilon = pow(10, -40);

void KNNModel::train(string filename) {

    this->features.clear();
    this->labels.clear();
    this->keys.clear();

    ifstream fileStream(filename);
    if (!fileStream.is_open()) {
        cout<<"Could not open " + filename + "file!\n";
        exit (1);
    }
    string line;
    int currentKey = -1;

    //skip headings
    getline(fileStream, line);

    while (getline(fileStream, line)) {
        vector<double> pixels;
        pixels.reserve(pixelCount);
        currentKey++;

        //split line by " " , "," "tab"
        char* chunk = strtok(const_cast<char*>(line.c_str()), " \t,");

        labels.push_back(atof(chunk));
        size_t counter = 0;
        if(pixelCount > 0) {
            while((chunk = strtok(nullptr, " \t,")) != nullptr && counter < pixelCount) {
                pixels.push_back(atof(chunk));
                counter++;
            }
        }
        else {
            while((chunk = strtok(nullptr," \t,")) != nullptr) {
                pixels.push_back(atof(chunk));
            }
        }
        this->features.push_back(pixels);
        this->keys.push_back(currentKey);
    }
    this->pixelCount = features[0].size();
    fileStream.close();
}

template<typename T>
static double EuclideanDistance(const vector<T>& left, const vector<T>& right) {

    if (left.size() != right.size()) {
        cout<<"vectors of different lengths passed to EuclideanDistance!\n"<<left.size()<<" "<<right.size()<<endl;
        exit(1);
    }
    double sum = 0;
    for(size_t index = 0; index < left.size(); index++) {
        sum += pow(left[index] - right[index], 2);
    }
    return pow(sum, 0.5);
}

bool compare(const pair<int,double>& left, const pair<int,double>& right) {
    return right.second - left.second > epsilon;
}

int predictLabel(vector<pair<int,double>>& sortedDistances, const vector<int>& labels, size_t K) {

    int prediction = 0;
    unordered_map<int,int> poll;
    for (size_t distIndex = 0; distIndex < K; distIndex++){
        int label = labels[sortedDistances[distIndex].first];
        if (poll.find(label) == poll.end()){
            poll[label] = 1;
        }
        else{
            poll[label]++;
        }
    }

    int counter = 0;
    for (const auto& iter: poll){
        if (iter.second > counter){
            prediction = iter.first;
            counter = iter.second;
        }
        /*
        else if(iter.second == counter){
            //TODO
        }
        */
    }
    return prediction;
}

vector<int> predictor(KNNModel* model, const vector<vector<double>>& predictFeatures, const vector<size_t> predictKeys, size_t K) {

    vector<int> predictLabels(predictKeys.size(), 0);
    const vector<vector<double>>& patrons = model->getFeatures();
    size_t patronsCount = patrons.size();
    vector<pair<int,double>> distances(patronsCount, {0,0});
    const vector<int>& labels = model->getLabels();

    for(const auto& key: predictKeys){
        const vector<double>& current = predictFeatures[key];
        for (size_t patronIndex = 0; patronIndex < patronsCount; patronIndex++){
            distances[patronIndex] = {patronIndex, EuclideanDistance(current, patrons[patronIndex])};
        }

        sort(distances.begin(), distances.end(), compare);
        predictLabels[key] = predictLabel(distances, labels, K);
    }
    return predictLabels;
}
//
vector<int> KNNModel::predict(string filename, size_t K) {

    vector<vector<double>> predictFeatures;
    vector<size_t> predictKeys;

    ifstream fileStream(filename);
    if (!fileStream.is_open()){
        cout<<"Could not open " + filename + "file!\n";
        exit (1);
    }

    string line;
    int currentKey = -1;

    //skip headings
    getline(fileStream,line);

    while (getline(fileStream,line)) {
        vector<double> pixels;
        pixels.reserve(pixelCount);
        currentKey++;

        //split line by " " , "," "tab"
        char* chunk = strtok(const_cast<char*>(line.c_str()), " \t,");
        size_t counter = 0;
        do {
            pixels.push_back(atof(chunk));
            counter++;
        }while((chunk = strtok(nullptr, " \t,")) != nullptr && counter < pixelCount);

        predictFeatures.push_back(pixels);
        predictKeys.push_back(currentKey);
    }
    fileStream.close();
    return predictor(this, predictFeatures, predictKeys, K);
}

void splitFeaturesForCrossValidate(KNNModel* model, vector<vector<vector<double>>>& featuresFolds, vector<vector<int>>& labelsFolds, vector<size_t>& keysCopy, size_t fold, size_t length, int fraction) {

    for(size_t counter = 0; counter < fold; counter++) {
        size_t from = counter * fraction;
        size_t to;
        if(counter == fold - 1) {
            to = length;
        }
        else {
            to = from + fraction;
        }
        vector<vector<double>> featuresToAdd;
        vector<int> labelsToAdd;
        featuresToAdd.reserve(fraction);
        labelsToAdd.reserve(fraction);
        for (size_t keyIndex = from; keyIndex < to; keyIndex++) {
            featuresToAdd.push_back(model->getFeatures()[keysCopy[keyIndex]]);
            labelsToAdd.push_back(model->getLabels()[keysCopy[keyIndex]]);
        }
        featuresFolds.push_back(featuresToAdd);
        labelsFolds.push_back(labelsToAdd);
    }
}

vector<double> calculateAccuracyForSubmodels(vector<vector<vector<double>>>& featuresFolds, vector<vector<int>>& labelsFolds, size_t pixelCount, size_t fold, size_t length, size_t K) {

    vector<double> accuracy;
    vector<vector<double>> trainFeatures;
    vector<int> trainLabels;
    vector<size_t> trainKeys;
    trainFeatures.reserve(length);
    trainLabels.reserve(length);
    trainKeys.reserve(length);
    for (size_t predictIndex = 0; predictIndex < fold; predictIndex++) {
        trainFeatures.clear();
        trainLabels.clear();
        trainKeys.clear();
        const vector<vector<double>>& predictFeatures = featuresFolds[predictIndex];
        const vector<int>& actualLabels = labelsFolds[predictIndex];
        vector<int> predictedLabels (actualLabels.size(),0);
        vector<size_t> predictKeys;
        for (size_t index = 0; index < actualLabels.size(); index++) {
            predictKeys.push_back(index);
        }

        int counter = -1;
        for (size_t trainIndex = 0; trainIndex < fold; trainIndex++) {
            if (trainIndex == predictIndex){
                continue;
            }
            size_t trainLength = featuresFolds[trainIndex].size();
            for (size_t it = 0; it < trainLength; it++) {
                trainFeatures.push_back(featuresFolds[trainIndex][it]);
                trainLabels.push_back(labelsFolds[trainIndex][it]);
                trainKeys.push_back(++counter);
            }
        }
        KNNModel submodel(trainFeatures, trainLabels, trainKeys, pixelCount);
        vector<int> results = predictor(&submodel, predictFeatures, predictKeys, K);

        double correct = 0;
        size_t total = predictKeys.size();
        for(size_t index = 0; index < total; index++) {
            if (predictedLabels[index] == actualLabels[index]) {
                correct++;
            }
        }
        double successRate = correct/total;
        accuracy.push_back(successRate);
    }
    return accuracy;
}

double KNNModel::crossValidate(size_t fold, size_t K) {

    size_t length = this->getFeatures().size();
    if (length == 0){
        cout<<"Train model before cross validating!\n";
        return 0;
    }

    int fraction = length / fold;
    vector<vector<vector<double>>> featuresFolds;
    vector<vector<int>> labelsFolds;
    vector<size_t> keysCopy(keys);

    splitFeaturesForCrossValidate(this, featuresFolds, labelsFolds, keysCopy, fold, length, fraction);
    vector<double> accuracy = calculateAccuracyForSubmodels(featuresFolds, labelsFolds, this->getPixelCount(), fold, length, K);

    double sum = 0;
    for(double sub : accuracy){
        sum += sub;
    }
    return sum / fold;
}
