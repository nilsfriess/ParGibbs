add_executable(rb_gibbs main.cc)
target_include_directories(rb_gibbs SYSTEM PRIVATE ../../external/)
target_include_directories(rb_gibbs SYSTEM PRIVATE "${pcgcpp_INCLDUE_DIR}")
target_compile_options(rb_gibbs PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(rb_gibbs pargibbs Eigen3::Eigen)

target_precompile_headers(rb_gibbs
  PRIVATE
  <chrono>
  <memory>
  <pcg_random.hpp>
)

