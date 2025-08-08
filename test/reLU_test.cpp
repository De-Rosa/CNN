#include <Eigen/Dense>
#include <layers/reLULayer.hpp>
#include <gtest/gtest.h>

Eigen::MatrixXd generateTestMatrix() {
	Eigen::MatrixXd mat(2,2);
	mat << 3, -1, -5, -1;
	return mat;
}

TEST(ReLUForwardPass, PositiveAndNegativeValues) {
	Eigen::MatrixXd mat = generateTestMatrix();

	ReLULayer layer;
	Eigen::MatrixXd result = layer.FeedForward(mat);

	Eigen::MatrixXd expected(2,2);
	expected << 3, 0, 0, 0;

	ASSERT_TRUE(result.isApprox(expected));
};

TEST(ReLUBackwardPass, PositiveAndNegativeValues) {
	Eigen::MatrixXd mat = generateTestMatrix();

	ReLULayer layer;
	Eigen::MatrixXd result = layer.FeedBackward(mat); 
	Eigen::MatrixXd expected(2,2);
	expected << 1, 0, 0, 0;

	ASSERT_TRUE(result.isApprox(expected));
}
