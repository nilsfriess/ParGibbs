include(FetchContent)
FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)

cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

add_executable(rb_gibbs main.cc)
target_include_directories(rb_gibbs PRIVATE ../../include)
target_include_directories(rb_gibbs SYSTEM PRIVATE ../../external/)
target_include_directories(rb_gibbs SYSTEM PRIVATE "${pcgcpp_INCLDUE_DIR}")
target_compile_options(rb_gibbs PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(rb_gibbs pargibbs Eigen3::Eigen)

