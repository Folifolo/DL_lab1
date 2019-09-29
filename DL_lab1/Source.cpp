#include "utils.h"
#include "Neural_Network.h"
#include <iostream>
#include <vector>
#include  <iomanip>

int main()
{
	int size = 3;
	int neurons = 10;
	double* ar = new double[784 * size];
	double* lb = new double[10* size];
	for (int i = 0; i < 10 * size; i++)
		lb[i] = 0;
	Load_Test_Data(ar, lb, size);
	for (int j = 0; j < 3; j++)
	for (int i = 0; i < 10; i++)
	if(lb[j*10+i]!=0)
		std::cout << "num " << i << endl;
	Neural_Network net(neurons, 0.001, 1000);
	net.Fit(ar, lb, size);

	//double* b1 = new double[neurons*size];
	//double* b2 = new double[10*size];
	//double* div1 = new double[neurons * size];
	//double* div2 = new double[10 * size];
	//double E;
	//double** dE1 = new double* [size];
	//double** dE2 = new double* [size];
	//for (int i = 0; i < size; i++)
	//{
	//	dE1[i] = new double[784 * neurons];
	//	dE2[i] = new double[10 * neurons];
	//}
	//Neural_Network net(neurons, 0.005, 100);
	//net.Forward(ar, b1, b2, div1, div2, size);
	//for (int j = 0; j < size; j++) {
	//	for (int i = 0; i < 10; i++)
	//		std::cout << std::setprecision(2) << b2[j*10 + i] << " ";
	//	std::cout << endl;
	//}
	////for (int i = 0; i < 10; i++)
	////	std::cout << div2[i] << "\n";

	//net.Calculate_E(ar, lb, E, dE1, dE2, size, b1, b2, div1, div2);
	//std::cout << "acc: " << net.Calculate_acc(lb, b2, size)<< endl;

	//for (int i = 0; i < 10; i++)
	//	if(lb[i]!=0)
	//		std::cout << "num " << i << endl;
	//
	////for (int i = 4*784; i < 5*784; i++)
	////{
	////	if (dE1[0][i] != 0)
	////	{
	////		std::cout << 1;
	////			//std::cout << i << " "<< dE1[0][i]<<endl;
	////			//break;
	////	}
	////	else
	////		std::cout << 0;
	////	if (i % 28 == 0)
	////		std::cout << endl;
	////}
	//std::cout << "err = " << E << endl;
	////for (int i = 0; i < 5; i++)
	////{
	////	for (int j = 0; j < 784; j++)
	////		std::cout << div2[i * 784 + j] << "  ";
	////	std::cout << "\n";
	////}
	//net.Back_Prop(dE1, dE2, size);
	//net.Forward(ar, b1, b2, div1, div2, size);
	//std::cout << "acc: " <<  net.Calculate_acc(lb, b2, size) << endl;
	//for (int j = 0; j < size; j++) {
	//	for (int i = 0; i < 10; i++)
	//		std::cout << std::setprecision(2) << b2[j * 10 + i] << " ";
	//	std::cout << endl;
	//}

	//net.Calculate_E(ar, lb, E, dE1, dE2, size, b1, b2, div1, div2);
	//std::cout << "err = " << E << endl;
	////for (int i = 0; i < 10; i++)
	////	std::cout << dE2[0][i] << "\n";


	////int count = 73;

	////for (int i = 0; i < 28; i++)
	////{
	////	for (int j = 0; j < 28; j++)
	////		std::cout << ar[count*784+(28 * i) + j];
	////	std::cout << "\n";
	////}
	////for (int i = 0; i < 10; i++)
	////	std::cout << lb[count*10+i];
	return 0;
}