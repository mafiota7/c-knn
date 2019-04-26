#include "KNNModel.cpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
using namespace std;

static void writeResults(string filename, const vector<int>& data){

    ofstream fileStream(filename);
    string header = "ImageId,Label\n";
    fileStream.write(header.c_str(), strlen(header.c_str()) * sizeof(char));

    size_t length = data.size();
    for (size_t index = 0; index < length; index++){
        string line = to_string(index + 1) + "," + to_string(data[index]) + "\n";
        fileStream.write(line.c_str(), strlen(line.c_str()) * sizeof(char));
    }
    fileStream.close();
}

int main() {

    string trainFilename ("train.csv");
    string unlabeledFilename ("test.csv");
    string outputFilename ("result.txt");
    size_t K = 3;
    size_t Nfold = 0;
    size_t components = 0;

    KNNModel M1 = KNNModel(components);
    M1.train(trainFilename);

    if (Nfold < 2) {
        vector<int> results = M1.predict(unlabeledFilename, K);
        writeResults(outputFilename, results);
    }
    else {
        double accuracy = M1.crossValidate(Nfold, K);
        ofstream fileStream(outputFilename);

        string line = "Fold = " + to_string(Nfold) + ", K = " + to_string(K) + ", accuracy = " + to_string(accuracy)+"\n";
        fileStream.write(line.c_str(), strlen(line.c_str()) * sizeof(char));

        fileStream.close();
    }
    return 0;
}
