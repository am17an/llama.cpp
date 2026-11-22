set(TARGET custom-op)
add_executable(${TARGET} custom-op.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# the CUDA kernel of the op is compiled to PTX, the example loads the file
if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUSTOM_OP_PTX ${CMAKE_CURRENT_BINARY_DIR}/custom-op.ptx)
    add_custom_command(
        OUTPUT  ${CUSTOM_OP_PTX}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -o ${CUSTOM_OP_PTX} ${CMAKE_CURRENT_SOURCE_DIR}/custom-op.cu
        DEPENDS custom-op.cu
        VERBATIM
    )
    add_custom_target(custom-op-ptx DEPENDS ${CUSTOM_OP_PTX})
    add_dependencies(${TARGET} custom-op-ptx)
    target_compile_definitions(${TARGET} PRIVATE CUSTOM_OP_PTX="${CUSTOM_OP_PTX}")
endif()
