add_executable(iact_test iact_test.cc)

target_include_directories(iact_test SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_compile_options(iact_test PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(iact_test parmgmc)
