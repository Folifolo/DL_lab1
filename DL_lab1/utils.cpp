#include <fstream>
#include <vector>
#include "utils.h"
#include "omp.h"
#include <iostream>


std::string path_to_data = ".\\MNIST";
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void Read_MNIST(string path, int NumberOfImages, int DataOfAnImage, double* arr)
{
	//arr = new double(NumberOfImages*DataOfAnImage);
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < NumberOfImages; ++i)
		{
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					arr[i*n_cols*n_rows + r*n_cols + c] = (double)temp/255;
				}
			}
		}
	}
}


void Read_MNIST_Label(string path,  int NumberOfImages, double * arr)
{
	//arr = new double(NumberOfImages);
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);	
		for (int i = 0; i < NumberOfImages; ++i)
		{			
			unsigned char temp = 0;
			file.read((char*)& temp, sizeof(temp));
			arr[i*10 + (int)temp] = 1;
		}
	}
}

void Load_Train_Data(double *images, double* lables, int size)
{
	std::string train_images_path = path_to_data + "\\train-images.idx3-ubyte";
	std::string train_labels_path = path_to_data + "\\train-labels.idx1-ubyte";
	Read_MNIST(train_images_path, size, 784, images);
	Read_MNIST_Label(train_labels_path, size, lables);
}

void Load_Test_Data(double* images, double* lables, int size)
{
	std::string test_images_path = path_to_data + "\\t10k-images.idx3-ubyte";
	std::string test_labels_path = path_to_data + "\\t10k-labels.idx1-ubyte";
	Read_MNIST(test_images_path, size, 784, images);
	Read_MNIST_Label(test_labels_path, size, lables);
}

void Multiplication_Array_Vector(double* Arr, double* Vec, double* Res, int size_i, int size_j)
{
	for (int i = 0; i < size_i; i++)
		Res[i] = 0.0;
	for (int i = 0; i < size_i; i++)
		for (int j = 0; j < size_j; j++)
			Res[i] += Arr[i * size_j + j] * Vec[j];
}

double Argmax(double* arr, int size)
{
	int argmax = 0;
	double max = 0;
	for (int i = 0; i < size; i++)
		if (arr[i] > max)
		{
			argmax = i;
			max = arr[i];
		}
	return argmax;
}
