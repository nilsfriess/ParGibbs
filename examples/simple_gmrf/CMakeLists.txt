include(FetchContent)
FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)
cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

add_executable(simple_gmrf main.cc)
target_include_directories(simple_gmrf SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/external")
target_include_directories(simple_gmrf SYSTEM PRIVATE "${pcgcpp_INCLDUE_DIR}")
target_compile_options(simple_gmrf PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(simple_gmrf pargibbs Eigen3::Eigen)

target_precompile_headers(simple_gmrf
  PRIVATE
  <chrono>
  <memory>
  <vector>
  <Eigen/Core>
  <Eigen/Eigen>
  <Eigen/Sparse>
)
