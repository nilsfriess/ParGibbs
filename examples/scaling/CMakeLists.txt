add_executable(scaling scaling.cc)

target_include_directories(scaling PRIVATE ../include)
target_include_directories(scaling SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})

target_compile_options(scaling PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(scaling parmgmc)
