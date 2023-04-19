cmake_minimum_required(VERSION 3.5) 
PROJECT (my_opencv_package)
find_package(OpenCV REQUIRED )
if (${OpenCV_FOUND})
  message("[MY_MESSAGE] OpenCV version: ${OpenCV_VERSION} found from the path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(FATAL_ERROR "OpenCV not found!")
endif()

set( SRC_FILES_PATH_HARRIS src/harrisCornerDetector.cpp src/filter.cpp src/helper_functions.cpp)
set( SRC_FILES_PATH_CALIBAR src/calibAR.cpp src/filter.cpp src/helper_functions.cpp src/csv_util.cpp src/modes.cpp)
set( SRC_FILES_PATH_SAMPLE_CODE src/sample_code.cpp)

# Debug using VSCode: https://code.visualstudio.com/docs/cpp/cmake-linux
# message("[MY_MESSAGE] CMake reading source files from: ${SRC_FILES_PATH_IMG_DISP}")
# Ref: https://stackoverflow.com/questions/13970377/configuring-an-c-opencv-project-with-cmake
set( INCLUDE_FILES_PATH       
    #  include/header.h
    include/
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} )
message("[MY_MESSAGE] CMake reading include files from: ${CMAKE_CURRENT_SOURCE_DIR}/include ; ${OpenCV_INCLUDE_DIRS}. Check INCLUDE_DIRECTORIES in CMakeLists.txt to double confirm")

link_directories( ${CMAKE_BINARY_DIR}/bin)
message("[MY_MESSAGE] CMake link_directories set to: ${CMAKE_BINARY_DIR}/bin")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(p04_harrisCornerDetector harrisCornerDetector)
add_executable( ${p04_harrisCornerDetector} ${SRC_FILES_PATH_HARRIS} )
target_link_libraries( ${p04_harrisCornerDetector} ${OpenCV_LIBS} )


set(p04_calibAR calibAR)
add_executable( ${p04_calibAR} ${SRC_FILES_PATH_CALIBAR} )
target_link_libraries( ${p04_calibAR} ${OpenCV_LIBS} )

set(p03_sample_code sample_code)
add_executable( ${p03_sample_code} ${SRC_FILES_PATH_SAMPLE_CODE} )
target_link_libraries( ${p03_sample_code} ${OpenCV_LIBS} )




# message("[MY_MESSAGE] Executable ${p02_write_csv} is linked against: ${OpenCV_LIBS}")
message("[MY_MESSAGE] Executable is saved to: ${EXECUTABLE_OUTPUT_PATH}/${p04_harrisCornerDetector}")
message("[MY_MESSAGE] Executable is saved to: ${EXECUTABLE_OUTPUT_PATH}/${p04_calibAR}")
