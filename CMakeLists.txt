cmake_minimum_required(VERSION 3.10)
project(CNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
include_directories(include/eigen)  

file(GLOB_RECURSE SRC_FILES
    src/*.cpp
)

add_executable(cnn ${SRC_FILES})

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(test)

