include_directories(.)

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
    )

# Gram-CTC loss
# ...


# Decoder
pybind11_add_module(cpp_ctc_decoder MODULE
    ${INNER_LIBRARIES}
    decoders/ctc_decoder.h
    decoders/ctc_decoder.cpp
    decoders/ctc_decoder_py.cpp)

target_link_libraries(cpp_ctc_decoder
    PRIVATE ${TORCH_LIBRARIES}
    PUBLIC kenlm
    )
