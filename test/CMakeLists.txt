file(GLOB TEST_FILES "*.cpp")

add_executable(run_tests ${TEST_FILES})
target_link_libraries(run_tests gtest gtest_main)
target_include_directories(run_tests PRIVATE ../include)

add_test(NAME ReLULayerTest COMMAND run_tests)

