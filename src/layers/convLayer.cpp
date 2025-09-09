#include "layers/convLayer.hpp"
#include "utils.hpp"

#include <stdexcept>

std::vector<Matrix3D> InitialiseFilters(int channelCount, int filterCount, int filterSize) {
        if (channelCount < 1) throw std::runtime_error("channel count smaller than 1");
        if (filterSize < 1) throw std::runtime_error("filter size smaller than 1");
        if (filterCount < 1) throw std::runtime_error("filter count fewer than 1");
    std::vector<Matrix3D> filters;
    filters.reserve(filterCount);

    for (int i = 0; i < filterCount; ++i) {
        Matrix3D filter;
        filter.reserve(channelCount);

        for (int c = 0; c < channelCount; ++c) {
            // https://quuxplusone.github.io/blog/2021/03/03/push-back-emplace-back/
            // factory method returns a prvalue that already calls move semantics so move is unnecessary
            filter.push_back(Matrix::Random(filterSize, filterSize));
        }

        filters.push_back(std::move(filter));
    }

    return filters;
}

// TODO: alter for loops for another solution
Matrix ApplyFilter(const Matrix& mat, const Matrix& filter, int stride) {
    int filterSize = filter.rows();
    auto [outputHeight, outputWidth] = ComputeDimsKernelValid(mat, filterSize, stride); 

    Matrix outputMatrix(outputHeight, outputWidth);
    auto filterArray = filter.array();

    for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
            // TODO: block may be inefficient, look into other views of matrix (or using tensors)
            auto mult = mat.block(stride * h, stride * w, filterSize, filterSize).array() * filterArray;
            outputMatrix(h, w) = mult.sum();
        }
    }
    
    return outputMatrix;
}

Matrix ApplyFilter(const Matrix3D& mat, const Matrix3D& filter, int stride) {
    int filterSize = filter[0].rows();
    auto [outputHeight, outputWidth] = ComputeDimsKernelValid(mat[0], filterSize, stride); 

    Matrix outputMatrix = Matrix::Zero(outputHeight, outputWidth);

    // iterate over channels and sum to make 2d matrix
    for (int c = 0; c < mat.size(); ++c) {
        outputMatrix += ApplyFilter(mat[c], filter[c], stride);
    }

    return outputMatrix;
}

Matrix3D ApplyFilters(const Matrix3D& mat, const std::vector<Matrix3D>& filters, int stride) {
    Matrix3D output;
    output.reserve(filters.size());

    // iterate over filters, each creating a feature map (channel)
    for (int f = 0; f < filters.size(); ++f) {
        output.push_back(ApplyFilter(mat, filters[f], stride));
    }

    return output;
}

ConvLayer::ConvLayer(int channelCount, int filterCount, int filterSize, int stride)
    : filters(InitialiseFilters(channelCount, filterCount, filterSize))
    , channelCount(channelCount)
    , stride(stride)
{}

Matrix3D ConvLayer::FeedForward(const Matrix3D& mat) const {
    return ApplyFilters(mat, filters, stride);
}
