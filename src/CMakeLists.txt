add_executable(first_order_locally_linear
               first_order_locally_linear.cc)

target_include_directories(first_order_locally_linear PRIVATE ../include)
target_compile_options(first_order_locally_linear PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)  
target_link_libraries(first_order_locally_linear pargibbs Eigen3::Eigen)
