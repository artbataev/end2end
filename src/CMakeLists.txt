include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

# inner libraries
set(INNER_LIBRARIES
    utils/threadpool.cpp
    utils/threadpool.h
    utils/math_utils.cpp
    utils/math_utils.h
    losses/forward_backward.h
    losses/forward_backward.cpp
    )


# CTC loss
pybind11_add_module(cpp_ctc_loss MODULE
    ${INNER_LIBRARIES}
    losses/ctc_loss.h
    losses/ctc_loss.cpp
    losses/ctc_loss_py.cpp)

target_link_libraries(cpp_ctc_loss
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
    )

# Gram-CTC loss
pybind11_add_module(cpp_gram_ctc_loss MODULE
    ${INNER_LIBRARIES}
    losses/gram_ctc_loss.h
    losses/gram_ctc_loss.cpp
    losses/gram_ctc_loss_py.cpp)

target_link_libraries(cpp_gram_ctc_loss
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
    )


# Decoder
pybind11_add_module(cpp_ctc_decoder MODULE
    ${INNER_LIBRARIES}
    decoders/ctc_decoder.h
    decoders/ctc_decoder.cpp
    decoders/ctc_decoder_py.cpp)

target_link_libraries(cpp_ctc_decoder
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
    PUBLIC kenlm
    )
