cmake_minimum_required(VERSION 3.16.3)
project(ocr)
#set(CMAKE_PREFIX_PATH "/usr/share/doc/tesseract-ocr")
# set(Tesseract_DIR "/usr/share/doc/tesseract-ocr")
set(CMAKE_CXX_STANDARD 11)
#Find Tesseract tesseract

find_package(PkgConfig REQUIRED)
pkg_search_module(TESSERACT REQUIRED tesseract)
pkg_search_module(LEPTONICA REQUIRED lept)
include_directories(${TESSERACT_INCLUDE_DIRS})
include_directories(${LEPTONICA_INCLUDE_DIRS})
link_directories( ${TESSERACT_LIBRARY_DIRS} )
link_directories( ${LEPTONICA_LIBRARY_DIRS} )
set(OpenCV_DIR /home/sid/opencv/build)


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/tessocr.cpp src/helper.cpp)

target_link_libraries(${PROJECT_NAME} tesseract lept
${Tesseract_LIBRARIES}
${Leptonica_LIBRARIES}
${OpenCV_LIBS}
)
