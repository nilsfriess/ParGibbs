Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.1
)

FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)
cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# add_executable(tests coloring.cc test.cc)
add_executable(tests 
  test.cc 
  multicolor_gibbs.cc
  helpers.cc
  dm_hierarchy.cc
  linear_operator.cc
  coloring.cc
  cholesky.cc
  mgmc.cc)
target_include_directories(tests SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_link_libraries(tests PRIVATE Catch2::Catch2 parmgmc)
catch_discover_tests(tests)
