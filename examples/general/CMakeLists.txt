add_executable(pcmg_test pcmg.cc)

target_include_directories(pcmg_test SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_include_directories(pcmg_test PRIVATE ../include)
target_compile_options(pcmg_test PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(pcmg_test parmgmc)
