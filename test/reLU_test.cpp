#include <Eigen/Dense>
#include <layers/reLULayer.hpp>
#include <gtest/gtest.h>

Eigen::MatrixXd generateReLUTestMatrix() {
	Eigen::MatrixXd mat(2,2);
	mat << 3, -1, -5, -1;
	return mat;
}

TEST(ReLUForwardPass, PositiveAndNegativeValues) {
	Matrix mat = generateReLUTestMatrix();

	ReLULayer layer;
	Matrix result = layer.FeedForward(mat);

	Matrix expected(2, 2);
	expected << 3, 0, 0, 0;

	ASSERT_TRUE(result.isApprox(expected));
};

TEST(ReLUBackwardPass, PositiveAndNegativeValues) {
	Matrix mat = generateReLUTestMatrix();
	Matrix grad(2,2);
	grad << 1, 1, 1, 1;

	ReLULayer layer;
	Matrix result = layer.FeedBackward(mat, grad); 
	Matrix expected(2, 2);
	expected << 1, 0, 0, 0;

	ASSERT_TRUE(result.isApprox(expected));
}
