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
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads)


# kenlm
find_package(Boost REQUIRED COMPONENTS
        program_options
        system
        thread
        unit_test_framework
        )
include(third_party/kenlm/cmake/KenLMFunctions.cmake)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(third_party/kenlm)
add_subdirectory(third_party/kenlm/util)
set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum supported ngram order")
set(KENLM_LM_SOURCE
        third_party/kenlm/lm/bhiksha.cc
        third_party/kenlm/lm/binary_format.cc
        third_party/kenlm/lm/config.cc
        third_party/kenlm/lm/lm_exception.cc
        third_party/kenlm/lm/model.cc
        third_party/kenlm/lm/quantize.cc
        third_party/kenlm/lm/read_arpa.cc
        third_party/kenlm/lm/search_hashed.cc
        third_party/kenlm/lm/search_trie.cc
        third_party/kenlm/lm/sizes.cc
        third_party/kenlm/lm/trie.cc
        third_party/kenlm/lm/trie_sort.cc
        third_party/kenlm/lm/value_build.cc
        third_party/kenlm/lm/virtual_interface.cc
        third_party/kenlm/lm/vocab.cc
        )

add_library(kenlm ${KENLM_LM_SOURCE} ${KENLM_LM_COMMON_SOURCE})
target_compile_definitions(kenlm PUBLIC -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})
target_link_libraries(kenlm kenlm_util ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


# inner libraries
add_library(threadpool
        pytorch_end2end/src/threadpool.cpp
        pytorch_end2end/src/threadpool.h)
target_link_libraries(threadpool
        ${CMAKE_THREAD_LIBS_INIT})

add_library(inner_math_utils
        pytorch_end2end/src/math_utils.cpp
        pytorch_end2end/src/math_utils.h)
target_link_libraries(inner_math_utils
        PRIVATE pybind11::module
        PRIVATE ${TORCH_LIBRARIES})


# CTC loss
add_library(cpp_ctc_loss MODULE
        pytorch_end2end/src/ctc_loss.cpp
        pytorch_end2end/src/ctc_loss.h)

target_link_libraries(cpp_ctc_loss
        PRIVATE threadpool
        PRIVATE inner_math_utils
        PRIVATE pybind11::module
        PRIVATE ${TORCH_LIBRARIES}
        )

set_target_properties(cpp_ctc_loss PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")


# Decoder
add_library(cpp_ctc_decoder MODULE
        pytorch_end2end/src/ctc_decoder.cpp
        pytorch_end2end/src/ctc_decoder.h)

target_link_libraries(cpp_ctc_decoder
        PRIVATE threadpool
        PRIVATE inner_math_utils
        PRIVATE pybind11::module
        PRIVATE ${TORCH_LIBRARIES}
        PUBLIC kenlm
        )

set_target_properties(cpp_ctc_decoder PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
