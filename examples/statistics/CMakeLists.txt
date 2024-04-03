add_executable(iact_test iact_test.cc)

target_include_directories(iact_test SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_include_directories(iact_test PRIVATE ../include)
target_compile_options(iact_test PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(iact_test parmgmc)

add_executable(convergence convergence.cc)

target_include_directories(convergence SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_include_directories(convergence PRIVATE ../include)
target_compile_options(convergence PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(convergence parmgmc)
