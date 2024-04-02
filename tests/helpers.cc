#include "parmgmc/common/helpers.hh"

#include <petscvec.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

using namespace Catch::Matchers;

TEST_CASE("fill_vec_rand fills vector with N(0,1) samples", "[.][long]") {
  Vec vec;
  const int size = 5;
  VecCreateSeq(MPI_COMM_WORLD, size, &vec);

  Vec mean;
  VecDuplicate(vec, &mean);
  VecZeroEntries(mean);

  Mat cov;
  MatCreateSeqDense(MPI_COMM_WORLD, size, size, nullptr, &cov);

  std::mt19937 engine;

  const int nSamples = 1000000;

  {
    for (int n = 0; n < nSamples; ++n) {
      // Function that takes size as parameter
      parmgmc::fillVecRand(vec, size, engine);

      VecAXPY(mean, 1. / nSamples, vec);
    }

    double norm;
    VecMean(mean, &norm);

    REQUIRE_THAT(norm, WithinAbs(0, 1e-3));
  }

  {
    for (int n = 0; n < nSamples; ++n) {
      // Function that doesn't take size as parameter
      parmgmc::fillVecRand(vec, engine);

      VecAXPY(mean, 1. / nSamples, vec);
    }

    double norm;
    VecMean(mean, &norm);

    REQUIRE_THAT(norm, WithinAbs(0, 1e-3));
  }
}
