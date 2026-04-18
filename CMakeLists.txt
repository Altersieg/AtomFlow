cmake_minimum_required(VERSION 3.24)  # 3.24+ required for complete CUDA LTO support
project(AtomFlow LANGUAGES CXX CUDA)

# ============================================================================
# 1. Project-level configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Release build (-O3)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Export compile_commands.json for clangd / IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# 2. CUDA architecture: target Ada Lovelace (sm_89)
# ----------------------------------------------------------------------------
# RTX 5060 Ti is Blackwell (sm_120), but the CUDA 12.x toolchain's PTX
# generation for sm_120 is not yet fully stable. We use the sm_89 code path
# for now: it already includes native FP8 E4M3 Tensor Core instructions
# (mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3) and the cvt.e4m3.*
# instructions required by Transformer Engine.
#
# Upgrade path: once the toolchain matures, switch to "89;120" to emit a
# fatbin that runs on both Ada and Blackwell cards.
# ============================================================================
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
message(STATUS "AtomFlow: targeting CUDA arch = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# 3. Build-mode switches (MVP vs peak-performance)
# ----------------------------------------------------------------------------
# ATOMFLOW_MONOLITHIC_BUILD = OFF (default, MVP):
#   CUDA_SEPARABLE_COMPILATION ON. Friendly to cross-TU template linking,
#   but disables aggressive cross-file inlining.
# ATOMFLOW_MONOLITHIC_BUILD = ON (peak):
#   Separable compilation OFF, CUDA LTO enabled. Embraces nvcc's
#   aggressive inlining across translation units.
# ============================================================================
option(ATOMFLOW_MONOLITHIC_BUILD
       "Disable separable compilation and enable CUDA LTO for peak performance"
       OFF)

option(ATOMFLOW_BUILD_TESTS  "Build unit tests (GoogleTest)"   OFF)
option(ATOMFLOW_BUILD_PYTHON "Build Python bindings (PyBind11)" OFF)
option(ATOMFLOW_ENABLE_NCCL  "Enable multi-GPU via NCCL"       OFF)

# ============================================================================
# 4. Dependencies
# ============================================================================
find_package(CUDAToolkit REQUIRED)   # provides CUDA::cudart, CUDA::cublas

if(ATOMFLOW_ENABLE_NCCL)
    # Reserved slot: multi-GPU communication
    find_package(NCCL REQUIRED)
endif()

if(ATOMFLOW_BUILD_PYTHON)
    # Reserved slot: Python frontend
    find_package(pybind11 CONFIG REQUIRED)
endif()

# ============================================================================
# 5. Main target: atomflow executable
# ============================================================================
file(GLOB_RECURSE ATOMFLOW_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"  # includes profiler.cpp
)

add_executable(atomflow ${ATOMFLOW_SOURCES})

target_include_directories(atomflow PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(atomflow PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# ----------------------------------------------------------------------------
# 5.1 Apply selected build mode
# ----------------------------------------------------------------------------
if(ATOMFLOW_MONOLITHIC_BUILD)
    message(STATUS "AtomFlow: MONOLITHIC build mode (LTO, no separable compilation)")
    set_target_properties(atomflow PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        INTERPROCEDURAL_OPTIMIZATION ON       # host-side LTO
    )
    # Device-side LTO (nvcc 12.x)
    target_compile_options(atomflow PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-dlto>
    )
    target_link_options(atomflow PRIVATE
        $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
else()
    message(STATUS "AtomFlow: MVP build mode (separable compilation ON)")
    set_target_properties(atomflow PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ----------------------------------------------------------------------------
# 5.2 Compile options
# ----------------------------------------------------------------------------
target_compile_options(atomflow PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math              # enable fast-math path for __expf etc.
        -Xcompiler=-Wall
        -lineinfo                    # preserve source line info for Nsight Compute
    >
)

# Release-only optimisation flags
target_compile_options(atomflow PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

# ============================================================================
# 6. Reserved subdirectories: Python bindings / Tests
# ============================================================================
if(ATOMFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ATOMFLOW_BUILD_PYTHON)
    add_subdirectory(python)  # future PyBind11 wrapper lives here
endif()

# ============================================================================
# 7. Configuration summary
# ============================================================================
message(STATUS "==========================================================")
message(STATUS "  AtomFlow build configuration")
message(STATUS "  ----------------------------------------------------------")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch         : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA toolkit      : ${CUDAToolkit_VERSION}")
message(STATUS "  Monolithic build  : ${ATOMFLOW_MONOLITHIC_BUILD}")
message(STATUS "  Build tests       : ${ATOMFLOW_BUILD_TESTS}")
message(STATUS "  Build Python      : ${ATOMFLOW_BUILD_PYTHON}")
message(STATUS "  Enable NCCL       : ${ATOMFLOW_ENABLE_NCCL}")
message(STATUS "==========================================================")
