add_executable(${APP} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${APP}/main.cpp)
target_link_libraries(${APP} ncnn ${OpenCV_LIBS})