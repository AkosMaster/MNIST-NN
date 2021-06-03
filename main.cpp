#include <iostream>
#include <vector>
#include <string>
#include <ctime>

double learningRate = 0.5;
double momentum = 0.5;

struct Connection {
    double weight;
    double delta;
};

class Node;
typedef std::vector<Node> Layer;

class Node {
public:
    Node(unsigned nextLayerSize, int _index);
    void feedForward(Layer& prevLayer);
    double output = 0;
    double gradient;
    int index;
    double activationFunction(double in);
    double activationFunctionDerivative(double in);
    double sumDOW(Layer& nextLayer);
    void outputGradients(double target);
    void hiddenGradients(Layer& nextLayer);
    void updateWeights(Layer& prevLayer);
private:
    std::vector<Connection> connections;
};

void Node::updateWeights(Layer& prevLayer) {
    for (int i = 0; i < prevLayer.size(); i++) {
        Node& n = prevLayer[i];
        double oldDelta = n.connections[index].delta;
        double newDelta =
            learningRate * n.output * gradient + momentum * oldDelta;

        //std::cout << "delta: " << newDelta << " grad: " << gradient << " ? " << (gradient == 0) << std::endl;

        n.connections[index].delta = newDelta;
        n.connections[index].weight += newDelta;
    }
}

double Node::activationFunction(double in) {
    
    return /*in / (1 + abs(in));*/tanh(in);
}

double Node::activationFunctionDerivative(double in) {
    return /*activationFunction(in) * (1 - activationFunction(in));*/ 1 - in*in;
}

double Node::sumDOW(Layer& nextLayer) {
    double sum = 0;

    

    for (int i = 0; i < nextLayer.size() - 1; i++) {
        //std::cout << "DOW: " << nextLayer[i].gradient << std::endl;
        //std::cout << "dbg1: " << std::to_string(i) << std::endl;
        sum += connections[i].weight * nextLayer[i].gradient;
    }
    
    return sum;
}

void Node::outputGradients(double target) {
    double delta = target - output;
    
    gradient = delta * Node::activationFunctionDerivative(output);
    //std::cout << "d: " << gradient << std::endl;
}

void Node::hiddenGradients(Layer& nextLayer) {
    double dow = sumDOW(nextLayer);
    gradient = dow * Node::activationFunctionDerivative(output);
    //std::cout << "d2: " << gradient << std::endl;
}

Node::Node(unsigned nextLayerSize, int _index) {
    index = _index;
    //std::cout << "New Node(" + std::to_string(nextLayerSize) + ")" << std::endl;

    for (int i = 0; i < nextLayerSize; i++) {
        connections.push_back(Connection());
        connections.back().weight = ((double)rand() / (RAND_MAX)); // random between 1 and 0...
    }
}

void Node::feedForward(Layer& prevLayer) {
    double sum = 0;

    for (int i = 0; i < prevLayer.size(); i++) {
       sum += prevLayer[i].output * prevLayer[i].connections[index].weight;
    }

    output = Node::activationFunction(sum/prevLayer.size());
}

class NNet {
public:
    NNet(std::vector<unsigned> &structure);
    void feedForward(std::vector<double>& inputs);
    double backProp(std::vector<double>& targets);
    void results(std::vector<double>& results);
    void finishBatch();

private:
    std::vector<Layer> layers;
};

void NNet::results(std::vector<double>& results) {
    results.clear();

    for (int i = 0; i < layers.back().size() - 1; i++) {
        results.push_back(layers.back()[i].output);
    }
}

NNet::NNet(std::vector<unsigned>& structure) {
    unsigned layerCount = structure.size();
    for (int i = 0; i < layerCount; i++) {
        layers.push_back(Layer());

        unsigned nextLayerSize = i == layerCount - 1 ? 0 : structure[i + 1];

        unsigned nodeCount = structure[i] + 1;
        for (int j = 0; j < nodeCount; j++) {
            layers.back().push_back(Node(nextLayerSize, j));
        }
        layers.back().back().output = 1;
    }
}

void NNet::feedForward(std::vector<double> &inputs) {
    for (int i = 0; i < inputs.size(); i++) {
        layers[0][i].output = inputs[i];
    }

    for (int i = 1; i < layers.size(); i++) {
        //std::cout << "Layer " << std::to_string(i) << ":";
        Layer& prevLayer = layers[i - 1];
        for (int j = 0; j < layers[i].size() - 1; j++) {
            //std::cout << " node " << std::to_string(j) << "; ";
            layers[i][j].feedForward(prevLayer);
        }
        //std::cout << std::endl;
    }
}

double NNet::backProp(std::vector<double>& targets) {
    Layer& outputLayer = layers.back();
    double error = 0;

    for (int i = 0; i < outputLayer.size()-1; i++) {
        double delta = targets[i] - outputLayer[i].output;
        error += delta * delta;
    }
    error /= outputLayer.size() - 1;
    error = sqrt(error);

    //std::cout << "Error: " << error << std::endl;

    for (int i = 0; i < outputLayer.size() - 1; i++) {
        outputLayer[i].outputGradients(targets[i]);
    }

    for (int i = layers.size() - 2; i > 0; i--) {
        Layer& layer = layers[i];
        Layer& nextLayer = layers[i + 1];

        for (int j = 0; j < layer.size(); j++) {
            layer[j].hiddenGradients(nextLayer);
        }
    }

    for (int i = layers.size() - 1; i > 0; i--) {
        Layer& layer = layers[i];
        Layer& prevLayer = layers[i - 1];

        for (int j = 0; j < layer.size() - 1; j++) {
            layer[j].updateWeights(prevLayer);
        }
    }

    return error;
}

#include <iostream>
#include <fstream>
using namespace std;

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void compressImage(std::vector<double>& image) {
    std::vector<double> new_img;
    for (int x = 0; x < 14; x++) {
        for (int y = 0; y < 14; y++) {
            int tl = x * 2 * 28 + y * 2;
            int tr = x * 2 * 28 + y * 2 + 1;
            int bl = x * 2 * 28 + y * 2 + 28;
            int br = x * 2 * 28 + y * 2 + 28 + 1;

            double avg = (image[tl] + image[tr] + image[bl] + image[br]) / 4;
            new_img.push_back(avg);
        }
    }
    image.clear();
    image = new_img;
}

#include <iostream>
#include <string>
#include <filesystem>


void ReadMNIST_Images(int NumberOfImages, int DataOfAnImage, vector<vector<double>>& arr)
{
    arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
    ifstream file(std::filesystem::current_path().string() + std::string("\\train-images.idx3-ubyte"), ios::binary); //C:\\Users\\Admin\\Desktop\\Iskola\\nyamvadt biosz\\projekt\\Biosz Projekt\\x64\\Release\\
    
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for (int i = 0; i < NumberOfImages; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    arr[i][(n_rows * r) + c] = ((double)temp/255);

                }
            }
        }
    }
    else {
        std::cout << "Unable to open images file" << std::endl;
        throw runtime_error("Unable to open images file");
    }
}

void ReadMNIST_Labels(int& number_of_labels, vector<vector<double>>& arr) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(std::filesystem::current_path().string() + std::string("\\train-labels.idx1-ubyte"), ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        arr.resize(number_of_labels, vector<double>(1));
        for (int i = 0; i < number_of_labels; i++) {
            char number = 0;
            file.read((char*)&number, 1);
            arr[i].clear();

            for (int j = 0; j < 10; j++) {
                arr[i].push_back(((int) j == (int)number ? 1.0 : 0 ));
            }
        }
    }
    else {
        std::cout << "Unable to open labels file" << std::endl;
        throw runtime_error("Unable to open labels file");
    }
}

int vectonum(std::vector<double> vec) {
    double mr = 0;
    int mc = -1;
    for (int c = 0; c < 10; c++) {
        if (vec[c] > mr) {
            mr = vec[c];
            mc = c;
        }
    }
    return mc;
}

void printMNISTImage(std::vector<double>& img) {
    for (int x = 0; x < 28; x++) {
        for (int y = 0; y < 28; y++) {
            double p = img[x * 28 + y];
            
            if (p < 0.2)
                std::cout << " ";
            else if (p < 0.7)
                std::cout << ".";
            else
                std::cout << "#";
            
        }
        std::cout << std::endl;
    }
}

int main()
{
    srand(time(NULL));
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.setf(std::ios::showpoint);

    std::vector<unsigned> structure = {784,20,10,10};
    NNet neuralNet(structure);

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;

    int label_num;
    ReadMNIST_Images(60000, 784, inputs);
    std::cout << inputs.size() << " images" << std::endl;
    ReadMNIST_Labels(label_num, targets);
    std::cout << targets.size() << " labels" << std::endl;

    int c = 0;
    double neterror = INT_MAX;

    int inputcount = 50000;
    int batchsize = 50000;
    int iteration = 0;
    while (neterror > 0.05 && iteration < 5) {
        double errorsum = 0;
        for (int i = 0; i < batchsize; i++) {

            int x = rand() % inputcount;
            neuralNet.feedForward(inputs[x]);
            errorsum += neuralNet.backProp(targets[x]);
        }
        
        neterror = errorsum / batchsize;
        c++;
        if (c % 1 == 0) {
            std::cout << ++iteration << ": " << std::endl;
            std::cout << "err: " << neterror << std::endl;
            int x1 = rand() % inputcount;
            neuralNet.feedForward(inputs[x1]);
        }
        
    }
    system("cls");
    std::cout << "---------- *** ----------" << std::endl;
    std::cout << "err: " << neterror << std::endl;
    std::cout << iteration << " iterations" << std::endl;

    int score = 0;
    for (int c = 0; c < 10000; c++) {
        int i = 50000 + c;//rand() % 10000 + 50000;
        neuralNet.feedForward(inputs[i]);

        std::vector<double> results;
        neuralNet.results(results);        

        //printMNISTImage(inputs[i]);
        int res = vectonum(results);
        int tar = vectonum(targets[i]);
        if (res == tar) {
            score++;
        }
        //std::cout << "result: " << res << " target: " << tar << std::endl;
    }
    std::cout << "correct out of 10,000: " << score << std::endl;
    std::cout << "avg. precision: " << ((float)score / 10000)*100 << "%" << std::endl;
}