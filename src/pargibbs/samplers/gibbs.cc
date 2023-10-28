#include "pargibbs/samplers/gibbs.hh"
#include "pargibbs/lattice/lattice.hh"

#include <pcg_random.hpp>

namespace pargibbs {
template class GibbsSampler<
    Lattice2D, Eigen::SparseMatrix<double, Eigen::RowMajor>, pcg64>;
template class GibbsSampler<
    Lattice2D, Eigen::SparseMatrix<double, Eigen::ColMajor>, pcg64>;

template class GibbsSampler<
    Lattice2D, Eigen::SparseMatrix<double, Eigen::RowMajor>, pcg32>;
template class GibbsSampler<
    Lattice2D, Eigen::SparseMatrix<double, Eigen::ColMajor>, pcg32>;
} // namespace pargibbs
