# Ajustez en fonction de votre version de CMake
cmake_minimum_required (VERSION 2.8)

# Nom du projet
project (featuresTest)
SET(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
message(STATUS "${CMAKE_MODULE_PATH}=${${CMAKE_MODULE_PATH}}")
include(LibFindMacros)
find_package (OpenCV REQUIRED)


message("----------- trying to find Jsoncpp-------------")
find_package(Jsoncpp)

if(Jsoncpp_FOUND)
    message("INFO: we found LibJsoncpp on your pc.")
    message(Jsoncpp_FOUND = ${Jsoncpp_FOUND})
    message(Jsoncpp_INCLUDE_DIR = ${Jsoncpp_INCLUDE_DIR}) 
    message(Jsoncpp_LIBRARY = ${Jsoncpp_LIBRARY})
else(Jsoncpp_FOUND)
    message("WARNING: we couldn't find LibJsoncpp on your pc. DLC is disabled.")
endif(Jsoncpp_FOUND)

#set(LIBS ${Jsoncpp_LIBRARY})

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Jsoncpp_PROCESS_INCLUDES Jsoncpp_INCLUDE_DIR)
set(Jsoncpp_PROCESS_LIBS Jsoncpp_LIBRARY)

# Exécutable "featuresTest", compilé à partir du fichier main.cpp
add_executable (featuresTest main.cpp)

# featuresTest sera linké avec les bibliothèques d'OpenCV
target_link_libraries (featuresTest ${OpenCV_LIBS})

include_directories(${Jsoncpp_INCLUDE_DIR})
target_link_libraries(featuresTest ${Jsoncpp_LIBRARIES})
