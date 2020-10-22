cmake_minimum_required(VERSION 3.8)

if ($ENV{TRAVIS_BUILD_NUMBER})
    set(PROJECT_VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})
else ()
    set(PROJECT_VERSION 0.0.1)
endif ()

project(pytorch_end2end VERSION ${PROJECT_VERSION} LANGUAGES CXX)

find_package(Torch REQUIRED)
add_subdirectory(third_party/pybind11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # for KenLM

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

add_subdirectory(src)
