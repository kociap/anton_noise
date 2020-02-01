cmake_minimum_required(VERSION 3.12)

project(anton_noise CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(anton_noise)

set(ANTON_PUBLIC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/anton/noise.hpp"
)
    
set(ANTON_PRIVATE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/noise.cpp"
)

target_sources(anton_noise
    PUBLIC ${ANTON_PUBLIC_FILES}
    PRIVATE ${ANTON_PRIVATE_FILES}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${ANTON_PRIVATE_FILES} ${ANTON_PUBLIC_FILES})

target_include_directories(anton_noise
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)