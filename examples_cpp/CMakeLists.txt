add_executable(testcc main.cc)

target_include_directories(testcc PRIVATE ../include)
target_compile_options(testcc PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(testcc parmgmc)
