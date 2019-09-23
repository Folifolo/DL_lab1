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

	void Forward(double* input, double* output1, double* output2, double* div1, double* div2);

	void Calculate_dE(double* input, double* label, double* dE1, double* dE2,
		double* output1, double* output2, double* div1);

	void Calculate_E(double* input, double* label, double& E, double* output2, int number_of_images);


	double Calculate_acc(double* label, double* output2, int number_of_images);

	void Back_Prop(double* dE1, double* dE2);


public:
	Neural_Network(int neurons, double spd, int epochs); 

	void Learn(double* input, double* label, int number_of_images);
};

#endif