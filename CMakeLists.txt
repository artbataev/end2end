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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads)

# kenlm
add_subdirectory(third_party/kenlm)
include_directories(third_party/kenlm)

add_library(threadpool
        pytorch_end2end/src/threadpool.cpp
        pytorch_end2end/src/threadpool.h)
target_link_libraries(threadpool
        ${CMAKE_THREAD_LIBS_INIT})

add_library(cpp_ctc_decoder MODULE
        pytorch_end2end/src/ctc_decoder.cpp
        pytorch_end2end/src/ctc_decoder.h)

add_library(cpp_ctc_loss MODULE
        pytorch_end2end/src/ctc_loss.cpp
        pytorch_end2end/src/ctc_loss.h)


target_link_libraries(cpp_ctc_decoder
        PRIVATE threadpool
        PRIVATE pybind11::module
        PRIVATE ${TORCH_LIBRARIES}
        PRIVATE kenlm
        )

set_property(TARGET cpp_ctc_decoder PROPERTY CXX_STANDARD 11)
set_target_properties(cpp_ctc_decoder PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_link_libraries(cpp_ctc_loss
        PRIVATE threadpool
        PRIVATE pybind11::module
        PRIVATE ${TORCH_LIBRARIES}
        )

set_property(TARGET cpp_ctc_loss PROPERTY CXX_STANDARD 11)
set_target_properties(cpp_ctc_loss PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
