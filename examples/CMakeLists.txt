cmake_minimum_required(VERSION 3.16)
project(stereo_tools CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

include_directories(/opt/nvidia/vpi3/include)
link_directories(/opt/nvidia/vpi3/lib/aarch64-linux-gnu)

add_executable(capture_pairs capture_pairs.cpp)
target_link_libraries(capture_pairs PRIVATE ${OpenCV_LIBS})

add_executable(stereo_calibrate stereo_calibrate.cpp)
target_link_libraries(stereo_calibrate PRIVATE ${OpenCV_LIBS})

add_executable(vpi_live_stereo vpi_live_stereo.cpp)
target_link_libraries(vpi_live_stereo PRIVATE ${OpenCV_LIBS} nvvpi)
