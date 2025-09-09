#ifndef UTILS_H
#define UTILS_H

#include <utility>

#include "layers/layer.hpp"

// Valid padding
inline std::pair<int, int> ComputeDimsKernelValid(const Matrix& mat, int size, int stride) {
    int height = ((mat.rows() - size) / stride) + 1;
	int width = ((mat.cols() - size) / stride) + 1;
    return std::pair(height, width);
}

#endif
