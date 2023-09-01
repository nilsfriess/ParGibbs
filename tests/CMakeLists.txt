enable_testing()

add_executable(
  tests
  samplers.cc
)

target_include_directories(tests PRIVATE ../src)

# find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(
  tests
  GTest::gtest_main
  Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(tests)
