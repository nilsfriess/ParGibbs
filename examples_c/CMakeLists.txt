add_executable(test main.c)

target_include_directories(test PRIVATE ../include)
target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(test parmgmc)
