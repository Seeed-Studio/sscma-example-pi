cmake_minimum_required(VERSION 3.9)

project(sscma)

find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

option(USE_SYSTEM_NCNN "build with system libncnn" OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ncnn.cmake)

macro(add_example name)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
endmacro()

add_example(yolov8)

