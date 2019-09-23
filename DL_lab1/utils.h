#ifndef UTILS_H
#define UTILS_H
#include <fstream>
#include <vector>
using namespace std;

int ReverseInt(int i);

void Read_MNIST(string path, int NumberOfImages, int DataOfAnImage, double*);

void Read_MNIST_Label(string path, int NumberOfImages, double*);

void Load_Train_Data(double* images, double* lables, int size);

void Load_Test_Data(double* images, double* lables, int size);

void  Multiplication_Array_Vector(double* Arr, double* Vec, double* Res, int size_i, int size_j);

double Argmax(double* arr, int size);

#endif