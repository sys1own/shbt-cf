cmake_minimum_required(VERSION 3.16)
project(shbt_cf_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(shbt_cf_sim src/main.cpp)

target_compile_options(shbt_cf_sim PRIVATE -Wall -Wextra -Wpedantic)
