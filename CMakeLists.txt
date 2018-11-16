cmake_minimum_required(VERSION 3.8)

if ($ENV{TRAVIS_BUILD_NUMBER})
    set(PROJECT_VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})
else ()
    set(PROJECT_VERSION 0.0.1)
endif ()

project(pytorch_end2end VERSION ${PROJECT_VERSION} LANGUAGES CXX)

find_package(Torch REQUIRED)
add_subdirectory(third_party/pybind11)

set(CMAKE_CXX_STANDARD 11)


# kenlm
add_subdirectory(third_party/kenlm)
include_directories(third_party/kenlm)


add_library(cpp_ctc_decoder MODULE pytorch_end2end/src/ctc_decoder.cpp
        pytorch_end2end/src/ctc_decoder.h
        pytorch_end2end/src/ctc_loss.cpp)


target_link_libraries(cpp_ctc_decoder
        PRIVATE pybind11::module
        ${TORCH_LIBRARIES}
        PRIVATE kenlm
        )

set_property(TARGET cpp_ctc_decoder PROPERTY CXX_STANDARD 11)
set_target_properties(cpp_ctc_decoder PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
