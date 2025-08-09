#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "network.hpp"
#include "layers/denseLayer.hpp"
#include "layers/reLULayer.hpp"
#include "optimisers/adam.hpp"
#include <gtest/gtest.h>

double MSELoss(const Matrix& pred, const Matrix& target) {
	return (pred - target).array().square().mean();
}

Matrix MSELossBackward(const Matrix& pred, const Matrix& target) {
	return 2.0 * (pred - target) / pred.rows();
}

double Train(Network& network, AdamOptimiser& adam, Matrix& inputs, Matrix& expected, int maxEpochs = 5000, int outputInterval = 500) {
	double loss = 0.0;

	for (int epoch = 1; epoch <= maxEpochs; epoch++) {
		// forward prop
		auto output = network.FeedForward(inputs);	
		loss = MSELoss(output, expected);

		// back prop
		auto loss_grad = MSELossBackward(output, expected);
		auto grad = network.FeedBackward(loss_grad);

		// update weights
		network.Optimise(adam);

		// zero gradients
		network.ZeroGradients();	

		if (epoch % outputInterval == 0) {
			std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
		}
	}

	return loss;
}

TEST(OptimiseTest, TrainsToFitLinearFunction) {
	Layers layers;
	layers.push_back(std::make_unique<DenseLayer>(1,10));
	layers.push_back(std::make_unique<ReLULayer>());
	layers.push_back(std::make_unique<DenseLayer>(10,1));
	Network network(std::move(layers));

	AdamOptimiser adam;

	// estimating function 2x + 3 
	int sampleCount = 100;
	Matrix inputs = Matrix::Random(sampleCount, 1) * 10; 
	Matrix expected = 2 * inputs.array() + 3;

	// train
	double loss = Train(network, adam, inputs, expected);

	// assert loss is low
	ASSERT_LT(loss, 0.01);

	Matrix testInputs(3,1);
	testInputs << -5, 0, 5;
	auto testOutput = network.FeedForward(testInputs);

	Matrix expectedOutputs(3,1);
	expectedOutputs << -7, 3, 13;

	std::cout << "Test output " << testOutput << std::endl;

	// predictions are within +-0.5
	ASSERT_TRUE(testOutput.isApprox(expectedOutputs, 0.5));
}

TEST(OptimiseTest, TrainsToFitQuadraticFunction) {
	Layers layers;
	layers.push_back(std::make_unique<DenseLayer>(1, 20));
	layers.push_back(std::make_unique<ReLULayer>());
	layers.push_back(std::make_unique<DenseLayer>(20, 1));
	Network network(std::move(layers));

	AdamOptimiser adam;

	// estimating function 3x^2 - 2x + 1
	int sampleCount = 200;
	Matrix inputs = Matrix::Random(sampleCount, 1) * 5;
	Matrix expected = 3 * inputs.array().square() - 2 * inputs.array() + 1;

	// train
	double loss = Train(network, adam, inputs, expected, 10000);

	// assert loss is low
	ASSERT_LT(loss, 0.075); 

	Matrix testInputs(5,1);
	testInputs << -3, -1, 0, 1, 3;
	auto testOutput = network.FeedForward(testInputs);

	Matrix expectedOutputs(5,1);
	expectedOutputs << 34, 6, 1, 2, 22;
	
	std::cout << "Test output " << testOutput << std::endl;

	// predictions are within +-0.5
	ASSERT_TRUE(testOutput.isApprox(expectedOutputs, 0.5));
}
