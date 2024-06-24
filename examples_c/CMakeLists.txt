add_executable(test main.c)

target_include_directories(test PRIVATE ../include)
target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(test parmgmc)


add_executable(matern matern.c)

target_include_directories(matern PRIVATE ../include)
target_compile_options(matern PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(matern parmgmc)


add_executable(fem fem.c)
