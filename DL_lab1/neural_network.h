#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

class Neural_Network
{
private:
	double* w1;
	double* w2;
	int input_size = 200;
	int hidden_size;
	int output_size;
	double speed;
	int iterations;
	void Init_Weights();

	double Relu(double x);

	double Relu_d(double x);

	void Forward(double* input, double* output1, double* output2, double* div1, double* div2);

	void Calculate_dE(double* input, double* label, double* dE1, double* dE2,
		double* output1, double* output2, double* div1);
		
	double Calculate_acc(double* label, double* output2, int number_of_images);

	void Back_Prop(double* dE1, double* dE2, int number_of_images);


public:
	Neural_Network(int neurons, double spd, int epochs); 

	void Fit(double* input, double* label, int number_of_images);

	void Fit_Batch(double* input, double* label, int number_of_images, int batch_size);

	void Predict(double* input, double* prediction, int number_of_images);

	double Calculate_E(double* label, double* prediction, int number_of_images);

};

#endif