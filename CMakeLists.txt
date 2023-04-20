cmake_minimum_required(VERSION 3.5) 
PROJECT (my_opencv_package)
find_package(OpenCV REQUIRED )
if (${OpenCV_FOUND})
  message("[MY_MESSAGE] OpenCV version: ${OpenCV_VERSION} found from the path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(FATAL_ERROR "OpenCV not found!")
endif()

set( SRC_FILES_PATH_YOLO src/yolo_v5.cpp  src/helper_functions.cpp)

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

set(prcv_final_project prcv_final_project)
add_executable( ${prcv_final_project} ${SRC_FILES_PATH_YOLO} )
target_link_libraries( ${prcv_final_project} ${OpenCV_LIBS} )




# message("[MY_MESSAGE] Executable ${p02_write_csv} is linked against: ${OpenCV_LIBS}")
message("[MY_MESSAGE] Executable is saved to: ${EXECUTABLE_OUTPUT_PATH}/${prcv_final_project}")
