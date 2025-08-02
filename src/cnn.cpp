#include <iostream>
#include <Eigen/Dense>
#include <layers/reLULayer.hpp>

int main (int argc, char *argv[]) {
	Eigen::MatrixXd mat(2,2);
	ReLULayer layer{};
	mat << 1,-1,-5,-1;
	std::cout << mat << std::endl;
	Eigen::MatrixXd mat2 = layer.FeedForward(mat); 
	std::cout << mat2 << std::endl;
	
	return 0;
}
