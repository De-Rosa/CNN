#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "network.hpp"
#include "layers/denseLayer.hpp"
#include "layers/reLULayer.hpp"
#include "optimisers/adam.hpp"
#include "optimisers/sgd.hpp"
#include <gtest/gtest.h>

double MSELoss(const Matrix& pred, const Matrix& target) {
	return (pred - target).array().square().mean();
}

Matrix MSELossBackward(const Matrix& pred, const Matrix& target) {
	return 2.0 * (pred - target) / pred.rows();
}

double Train(Network& network, Optimiser& optimiser, Matrix& inputs, Matrix& expected, int maxEpochs = 5000) {
	double loss = 0.0;

	for (int epoch = 1; epoch <= maxEpochs; epoch++) {
		// forward prop
		auto output = network.FeedForward(inputs);	
		loss = MSELoss(output, expected);

		// back prop
		auto loss_grad = MSELossBackward(output, expected);
		auto grad = network.FeedBackward(loss_grad);

		// update weights
		network.Optimise(optimiser);

		// zero gradients
		network.ZeroGradients();	
	}

	return loss;
}

Network CreateTestNetwork(int hiddenSize = 10) {
	Layers layers;
	layers.push_back(std::make_unique<DenseLayer>(1, hiddenSize));
	layers.push_back(std::make_unique<ReLULayer>());
	layers.push_back(std::make_unique<DenseLayer>(hiddenSize, 1));
	return Network(std::move(layers));
}

TEST(OptimiseTestAdam, TrainsToFitLinearFunction) {
	Network network = CreateTestNetwork();	
	AdamOptimiser adam;

	// estimating function 2x + 3 
	int sampleCount = 100;
	Matrix inputs = Matrix::Random(sampleCount, 1) * 10; 
	Matrix expected = 2 * inputs.array() + 3;

	// train
	double loss = Train(network, adam, inputs, expected);
	std::cout << "Adam Loss (Linear): " << loss << std::endl;

	// assert loss is low
	ASSERT_LT(loss, 0.01);

	Matrix testInputs(3,1);
	testInputs << -5, 0, 5;
	auto testOutput = network.FeedForward(testInputs);

	Matrix expectedOutputs(3,1);
	expectedOutputs << -7, 3, 13;

	// predictions are within +-0.5
	ASSERT_TRUE(testOutput.isApprox(expectedOutputs, 0.5));
}


TEST(OptimiseTestSGD, TrainsToFitLinearFunction) {
	Network network = CreateTestNetwork();	
	SGDOptimiser sgd;

	// estimating function 2x + 3 
	int sampleCount = 100;
	Matrix inputs = Matrix::Random(sampleCount, 1) * 10; 
	Matrix expected = 2 * inputs.array() + 3;

	// train
	double loss = Train(network, sgd, inputs, expected);
	std::cout << "SGD Loss (Linear): " << loss << std::endl;

	// assert loss is low
	ASSERT_LT(loss, 0.01);

	Matrix testInputs(3, 1);
	testInputs << -5, 0, 5;
	auto testOutput = network.FeedForward(testInputs);

	Matrix expectedOutputs(3, 1);
	expectedOutputs << -7, 3, 13;

	// predictions are within +-0.5
	ASSERT_TRUE(testOutput.isApprox(expectedOutputs, 0.5));
}

TEST(OptimiseTestAdam, TrainsToFitQuadraticFunction) {
	Network network = CreateTestNetwork(20);
	AdamOptimiser adam;

	// estimating function 3x^2 - 2x + 1
	int sampleCount = 200;
	Matrix inputs = Matrix::Random(sampleCount, 1) * 5;
	Matrix expected = 3 * inputs.array().square() - 2 * inputs.array() + 1;

	// train
	double loss = Train(network, adam, inputs, expected, 10000);
	std::cout << "Adam Loss (Quadratic): " << loss << std::endl;

	// assert loss is low
	ASSERT_LT(loss, 0.075); 

	Matrix testInputs(5, 1);
	testInputs << -3, -1, 0, 1, 3;
	auto testOutput = network.FeedForward(testInputs);

	Matrix expectedOutputs(5, 1);
	expectedOutputs << 34, 6, 1, 2, 22;
	
	// predictions are within +-0.5
	ASSERT_TRUE(testOutput.isApprox(expectedOutputs, 0.5));
}
