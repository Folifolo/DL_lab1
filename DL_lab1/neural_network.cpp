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
			w1[i * input_size + j] =  (double)rand() / (RAND_MAX);
	for (int i = 0; i < Neural_Network::output_size; i++)
		for (int j = 0; j < Neural_Network::hidden_size; j++)
			w2[i * hidden_size + j] = (double)rand() / (RAND_MAX);
}

Neural_Network::Neural_Network(int neurons, double spd, int epochs, int input_s, int output_s)
{
	//srand(time(0));
	Neural_Network::input_size = input_s;
	Neural_Network::output_size = output_s;
	Neural_Network::hidden_size = neurons;
	Neural_Network::w1 = new double[input_size * neurons];
	Neural_Network::w2 = new double[output_size * neurons];
	Neural_Network::Init_Weights();
	Neural_Network::speed = spd;
	Neural_Network::iterations = epochs;
}

double Neural_Network::Relu(double x)
{
	if (x < 0)
		x = 0;
	return x;
}

double Neural_Network::Relu_d(double x)
{
	double res = 1;
	if (x < 0)
		res = 0;
	return res;
}
void Neural_Network::Forward(double* input, double* output1, double* output2)
{
	Multiplication_Array_Vector(w1, input, output1, hidden_size, input_size);

	Multiplication_Array_Vector(w2, output1, output2, output_size, hidden_size);

	double out2_max = *std::max_element(output2, output2 + output_size);
	double sum = 0;

	for (int i = 0; i < output_size; i++)
	{
		sum += exp(output2[i] - out2_max);
	}

	for (int i = 0; i < output_size; i++)
	{
		output2[i] = exp(output2[i] - out2_max) / sum;
	}
}

void Neural_Network::Forward_div(double* input, double* output1, double* output2, double* div1, double* div2)
{
	for (int i = 0; i < hidden_size; i++)
	{
		div1[i] = 1;
	}

	for (int i = 0; i < output_size; i++)
	{
		div2[i] = output2[i] * (1 - output2[i]);
	}
}

void Neural_Network::Calculate_dE(double* input, double* label, double* dE1, double* dE2,
	double* output1, double* output2, double* div1)
{
	double dE1_sum;
	double reg_rate = 0;// 0.0001;

	for (int j = 0; j < output_size; j++)
		for (int s = 0; s < hidden_size; s++)
		{
			dE2[j * hidden_size + s] = -label[j] * (1 - output2[j]) * output1[s]
				+ reg_rate * (w2[j * hidden_size + s]* w2[j * hidden_size + s]);
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

double Neural_Network::Calculate_E(double* label, double* prediction, int number_of_images)
{
	double j_sum;
	double E = 0;
	for (int k = 0; k < number_of_images; k++)
	{
		j_sum = 0;
		for (int j = 0; j < output_size; j++)
		{
			if(prediction[k * output_size + j] != 0)
				j_sum += log(prediction[k*output_size + j]) * label[k * output_size + j];
		}
		E += j_sum;

	}
	E = -E/number_of_images;
	return E;
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
void Neural_Network::Back_Prop(double* dE1, double* dE2, int number_of_images)
{

	for (int j = 0; j < output_size; j++)
	{
		for (int s = 0; s < hidden_size; s++)
		{
			w2[j * hidden_size + s] -= (speed * dE2[j * hidden_size + s])/number_of_images;
		}
	}

	for (int s = 0; s < hidden_size; s++)
	{
		for (int i = 0; i < input_size; i++)
		{
					
			w1[s * input_size + i] -= (speed *0.8* dE1[s * input_size + i]) / number_of_images;

		}
	}
}

void Neural_Network::Fit(double* input, double* label, int number_of_images)
{

	double* b1 = new double[hidden_size * number_of_images];
	double* b2 = new double[output_size * number_of_images];
	double* div1 = new double[hidden_size * number_of_images];
	double* div2 = new double[output_size * number_of_images];
	double E;
	double* dE1 = new double [input_size * hidden_size];
	double* dE2 = new double [output_size * hidden_size];

	for (int epoch = 0; epoch < iterations; epoch++)
	{
		for (int image = 0; image < number_of_images; image++)
		{
			Forward(&input[image * input_size], &b1[hidden_size * image], &b2[output_size * image]);
			Forward_div(&input[image* input_size], &b1[hidden_size * image], &b2[output_size * image], div1, div2);

			Calculate_dE(&input[image * input_size], &label[image * output_size], dE1, dE2, &b1[hidden_size * image], &b2[output_size *image], div1);

			Back_Prop(dE1, dE2, 1);

		}
		Predict(input, b2, number_of_images);
		std::cout << epoch << " ";
		std::cout << "acc: " <<Calculate_acc(label, b2, number_of_images)<< ", ";
		E = Calculate_E(label, b2, number_of_images);
		std::cout << "err = " << E << endl;
		if (E != E)
			cout << "error is NaN ";
		std::cout << endl;
	}
}

//void Neural_Network::Fit_Batch(double* input, double* label, int number_of_images, int batch_size)
//{
//
//	double* b1 = new double[hidden_size * batch_size];
//	double* b2 = new double[output_size * number_of_images];
//	double* div1 = new double[hidden_size * batch_size];
//	double* div2 = new double[output_size * batch_size];
//	double E;
//	double* dE1 = new double[input_size * hidden_size];
//	double* dE2 = new double[output_size * hidden_size];
//	double* _dE1 = new double[input_size * hidden_size];
//	double* _dE2 = new double[output_size * hidden_size];
//
//	for (int j = 0; j < output_size; j++)
//		for (int s = 0; s < hidden_size; s++)
//			dE2[j * hidden_size + s] =0;
//	
//	for (int s = 0; s < hidden_size; s++)
//		for (int i = 0; i < input_size; i++)
//			dE1[s * input_size + i] = 0;
//
//	for (int epoch = 0; epoch < iterations; epoch++)
//	{
//		for (int batch = 0; batch < (int)number_of_images/batch_size; batch++)
//		{
//			for (int image = 0; image < batch_size; image++)
//			{
//				Forward(&input[(batch*batch_size + image) * input_size], &b1[hidden_size * image], &b2[output_size * image],
//					&div1[hidden_size * image], &div2[output_size * image]);
//
//				Calculate_dE(&input[(batch * batch_size + image) * input_size], &label[(batch * batch_size + image) * output_size],
//					_dE1, _dE2, &b1[hidden_size * image], &b2[output_size * image], &div1[hidden_size * image]);
//				for (int j = 0; j < output_size; j++)
//					for (int s = 0; s < hidden_size; s++)
//						dE2[j * hidden_size + s] += _dE2[j * hidden_size + s]/batch_size;
//
//
//				for (int s = 0; s < hidden_size; s++)
//					for (int i = 0; i < input_size; i++)
//						dE1[s * input_size + i] += _dE1[s * input_size + i] / batch_size;
//			}
//
//				Back_Prop(dE1, dE2, number_of_images);
//		}
//		Predict(input, b2, number_of_images);
//		std::cout << epoch << " ";
//		std::cout << "acc: " << Calculate_acc(label, b2, number_of_images) << ", ";
//		E = Calculate_E(label, b2, number_of_images);
//		std::cout << "err = " << E << endl;
//		if (E != E)
//			cout << "error is NaN ";
//		std::cout << endl; 
//		
//	}
//}

void Neural_Network::Predict(double* input, double* prediction, int number_of_images)
{
	double* b1 = new double[hidden_size];
	for (int image = 0; image < number_of_images; image++)
		Forward(&input[image * input_size], b1, &prediction[output_size * image]);
}