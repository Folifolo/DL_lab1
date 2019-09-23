#include "Neural_Network.h"
#include "utils.h"
#include <iostream>
# include <cmath>
#include  <iomanip>
#include <time.h>
#include <algorithm>

void Neural_Network::Init_Weights()
{
	for (int i = 0; i < Neural_Network::hidden_size; i++)
		for (int j = 0; j < Neural_Network::input_size; j++)
			w1[i * input_size + j] =  (double)rand() / (5*RAND_MAX);
	for (int i = 0; i < Neural_Network::output_size; i++)
		for (int j = 0; j < Neural_Network::hidden_size; j++)
			w2[i * hidden_size + j] = (double)rand() / (5 * RAND_MAX);
}

Neural_Network::Neural_Network(int neurons, double spd, int epochs)
{
	srand(time(0));
	Neural_Network::input_size = 784;
	Neural_Network::output_size = 10;
	Neural_Network::hidden_size = neurons;
	Neural_Network::w1 = new double[784 * neurons];
	Neural_Network::w2 = new double[10 * neurons];
	Neural_Network::Init_Weights();
	Neural_Network::speed = spd;
	Neural_Network::iterations = epochs;
}

void Neural_Network::Forward(double* input, double* output1, double* output2, double* div1, double* div2)
{

	double eps = 1e-90;
	Multiplication_Array_Vector(w1, input, output1, hidden_size, input_size);

	for (int i = 0; i < hidden_size; i++)
	{
		//if (abs(output1[i]) < eps)
		//	output1[i] = 0;
		div1[i] = 1;
	}

	Multiplication_Array_Vector(w2, output1, output2, output_size, hidden_size);

	//std::cout << *std::max_element(w2, w2 + hidden_size*10) << " ";
	double out2_max = *std::max_element(output2, output2 + output_size);
	double sum = 0;
	for (int i = 0; i < output_size; i++)
	{
		//std::cout << output2[i] << " ";
		sum += exp(output2[i]- out2_max);
		if (std::isinf(sum))
			cout << i;
	}

	for (int i = 0; i < output_size; i++)
	{
		output2[i] = exp(output2[i]- out2_max) / sum;
		
		div2[i] = output2[i] * (1 - output2[i]);
	}


}

void Neural_Network::Calculate_dE(double* input, double* label, double* dE1, double* dE2,
	double* output1, double* output2, double* div1)
{
	double dE1_sum;
	double reg_rate = 0;

	for (int j = 0; j < output_size; j++)
	{
		for (int s = 0; s < hidden_size; s++)
		{
			dE2[j * hidden_size + s] = -label[j] * (1 - output2[j]) * output1[s] +reg_rate * (w2[j * hidden_size + s]* w2[j * hidden_size + s]);
		}
	}

	for (int s = 0; s < hidden_size; s++)
		for (int i = 0; i < input_size; i++)
		{
			dE1_sum = 0;
			for (int j = 0; j < output_size; j++)
				dE1_sum += label[j] * (1 - output2[j]) * w2[j * hidden_size + s] * div1[s] * input[i];
			dE1[s * input_size + i] = -dE1_sum + reg_rate *(w1[s * input_size + i]* w1[s * input_size + i]);
		}
}

void Neural_Network::Calculate_E(double* input, double* label, double& E, double* output2, int number_of_images)
{
	double j_sum;
	E = 0;
	for (int k = 0; k < number_of_images; k++)
	{
		j_sum = 0;
		for (int j = 0; j < output_size; j++)
		{
			j_sum += log(output2[k*output_size + j]) * label[k * output_size + j];
		}
		E += j_sum;
	}
	E = -E;
}

double Neural_Network::Calculate_acc(double* label, double* output2, int number_of_images)
{
	double acc=0;
	for (int k = 0; k < number_of_images; k++)
	{
		if(Argmax(&label[k* output_size],output_size) == Argmax(&output2[k * output_size], output_size))
		acc+=1.0;
	}
	acc /= number_of_images;

	return acc;
}
void Neural_Network::Back_Prop(double* dE1, double* dE2)
{

	for (int j = 0; j < output_size; j++)
	{
		for (int s = 0; s < hidden_size; s++)
		{
			//if (dE2[k][j * hidden_size + s] != 0)
			//	cout << j << ": "<< w2[j * hidden_size + s] << " ";
			//w2[j * hidden_size + s] -= speed * dE2[k][j * hidden_size + s];
			//if (dE2[k][j * hidden_size + s] != 0)
			//	cout << w2[j * hidden_size + s] << endl;
			w2[j * hidden_size + s] -= speed*1.2 * dE2[j * hidden_size + s];
		}
	}

	for (int s = 0; s < hidden_size; s++)
	{
		for (int i = 0; i < input_size; i++)
		{
					
			w1[s * input_size + i] -= speed * dE1[s * input_size + i];

			//if (dE1[k][s * input_size + i])
			//	cout << 1 << " ";
			//else
			//	cout << 0 << " ";
			//if (i % 28 == 0)
			//	cout << endl;
					

			//if (abs(w1[s * input_size + i]>1))
				//cout << w1[s * input_size + i] << " ";
		}
		//cout << endl;
	}
}

void Neural_Network::Learn(double* input, double* label, int number_of_images)
{

	double* b1 = new double[hidden_size * number_of_images];
	double* b2 = new double[output_size * number_of_images];
	double* div1 = new double[hidden_size * number_of_images];
	double* div2 = new double[output_size * number_of_images];
	double E;
	double* dE1 = new double [784 * hidden_size];
	double* dE2 = new double [10 * hidden_size];

	for (int epoch = 0; epoch < iterations; epoch++)
	{
		for (int image = 0; image < number_of_images; image++)
		{
			//for (int i = 0; i < 784; i++)
			//{
			//	if (input[image * 784+i] != 0)
			//	{
			//		std::cout << 1;
			//			//std::cout << i << " "<< dE1[0][i]<<endl;
			//			//break;
			//	}
			//	else
			//		std::cout << 0;
			//	if (i % 28 == 0)
			//		std::cout << endl;
			//}
			Forward(&input[image*784], &b1[hidden_size * image], &b2[output_size * image], div1, div2);
			//std::cout << "before: ";
			//for (int i = 0; i < 10; i++)
			//	//if (i == 2 || i == 7)
			//		std::cout << std::setprecision(2) << b2[output_size * image+i] << " ";
			//std::cout << endl;

			Calculate_dE(&input[image * 784], &label[image * 10], dE1, dE2, &b1[hidden_size * image], &b2[output_size *image], div1);

			Back_Prop(dE1, dE2);
			Forward(&input[image * 784], &b1[hidden_size * image], &b2[output_size * image], div1, div2);
			//std::cout << "after: ";
			//for (int i = 0; i < 10; i++)
			//	//if (i == 2 || i == 7)
			//		std::cout << std::setprecision(2) << b2[output_size * image + i] << " ";

			//std::cout << endl;
		}
		std::cout << epoch << " ";
		std::cout << "acc: " <<Calculate_acc(label, b2, number_of_images)<< ", ";
		Calculate_E(input, label, E, b2, number_of_images);
		std::cout << "err = " << E << endl;
		if (E != E)
			cout << " ";
		std::cout << endl;
	}
}