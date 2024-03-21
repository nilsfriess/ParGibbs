add_executable(strong_scaling strong.cc)

target_include_directories(strong_scaling SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_compile_options(strong_scaling PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(strong_scaling parmgmc)
