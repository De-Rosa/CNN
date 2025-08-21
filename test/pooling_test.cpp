#include <Eigen/Dense>
#include <layers/poolingLayer.hpp>
#include <gtest/gtest.h>

Eigen::MatrixXd generatePoolTestMatrix() {
	Eigen::MatrixXd mat(4,4);
	mat << 7, 3, 5, 2, 8, 7, 1, 6, 4, 9, 3, 9, 0, 8, 4, 5;
	return mat;
}

TEST(MaxPoolForwardPass, PositiveValues) {
	Eigen::MatrixXd mat = generatePoolTestMatrix();

	MaxPoolingLayer layer(2, 2);
	Eigen::MatrixXd result = layer.FeedForward(mat);

	Eigen::MatrixXd expected(2, 2);
	expected << 8, 6, 9, 9;

	ASSERT_TRUE(result.isApprox(expected));
};

TEST(MaxPoolBackwardPass, PositiveValues) {
	Eigen::MatrixXd mat = generatePoolTestMatrix();
	Eigen::MatrixXd grad(2, 2);
	grad << 1, 2, 3, 4;

	MaxPoolingLayer layer(2, 2);
	Eigen::MatrixXd result = layer.FeedBackward(mat, grad);

	Eigen::MatrixXd expected(4, 4);
	expected << 0, 0, 0, 0, 1, 0, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0;

	ASSERT_TRUE(result.isApprox(expected));
}
