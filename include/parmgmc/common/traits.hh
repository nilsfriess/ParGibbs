#pragma once

#include <Eigen/Sparse>

#include <type_traits>

namespace parmgmc::detail {
template <typename> struct is_eigen_sparse_vector : std::false_type {};

template <typename T, int O, typename S>
struct is_eigen_sparse_vector<Eigen::SparseVector<T, O, S>> : std::true_type {};

template <typename... T>
inline constexpr bool is_eigen_sparse_vector_v =
    is_eigen_sparse_vector<T...>::value;
}; // namespace parmgmc::detail
