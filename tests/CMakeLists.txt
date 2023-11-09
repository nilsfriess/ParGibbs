include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  tests
  test_main.cc
  samplers.cc
  lattice.cc
  helpers.cc
)

target_include_directories(tests PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(
  tests
  pargibbs
  GTest::gtest_main
)
