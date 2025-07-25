cmake_minimum_required(VERSION 3.25)
project(sim2sim)

set(CMAKE_CXX_STANDARD 17)

# 添加LCM库
set(LCM_DIR "/usr/local/lib/lcm/cmake")
find_package(lcm REQUIRED)

set(LCM_INCLUDE_DIRS "/usr/local/include")
include_directories(${LCM_INCLUDE_DIRS})
set(LCM_LIBRARY_DIRS "/usr/local/lib")
link_directories(${LCM_LIBRARY_DIRS})

# 找到 CUDA
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.8)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-12.8/include")
set(CUDA_LIBRARY_DIRS "/usr/local/cuda-12.8/lib64")
set(cuBLAS_LIBRARY_DIRS "/usr/lib/x86_64-linux-gnu")
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${cuBLAS_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# 设置 LibTorch 的路径
set(Torch_DIR "/home/rot/anaconda3/envs/Tinker/lib/python3.8/site-packages/torch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "/home/rot/anaconda3/envs/Tinker/lib/python3.8/site-packages/torch/share/cmake/Torch")  # 将此路径替换为您解压 LibTorch 的路径
find_package(Torch REQUIRED)
include_directories(${Torch_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/lcm_types)
include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories("lcm_types/my_lcm/Request.hpp")
include_directories("lcm_types/my_lcm/Response.hpp")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/lcm_types/my_lcm/request.hpp
    OUTPUT ${CMAKE_BINARY_DIR}/lcm_types/my_lcm/response.hpp
    COMMAND lcm-gen -x ${CMAKE_SOURCE_DIR}/lcm_types/request.lcm
    COMMAND lcm-gen -x ${CMAKE_SOURCE_DIR}/lcm_types/response.lcm
    COMMAND lcm-gen -p ${CMAKE_SOURCE_DIR}/lcm_types/request.lcm
    COMMAND lcm-gen -p ${CMAKE_SOURCE_DIR}/lcm_types/response.lcm
    DEPENDS ${CMAKE_SOURCE_DIR}/lcm_types/request.lcm
    DEPENDS ${CMAKE_SOURCE_DIR}/lcm_types/response.lcm
    COMMENT "Generating LCM types"
)

add_custom_target(lcm_types ALL
    DEPENDS ${PROJECT_SOURCE_DIR}/lcm_types/my_lcm/Request.hpp
    DEPENDS ${PROJECT_SOURCE_DIR}/lcm_types/my_lcm/Response.hpp
)

add_executable(lcm_service src/lcm_service.cpp)
add_executable(lcm_publisher src/lcm_publisher.cpp)
add_executable(udp_publisher src/udp_publisher.cpp)
add_executable(udp_publisher_tinker src/udp_publisher_tinker.cpp)

add_dependencies(lcm_service lcm_types)
add_dependencies(lcm_publisher lcm_types)
add_dependencies(udp_publisher lcm_types)
add_dependencies(udp_publisher lcm_types)
set(LCM_LIBRARIES "/usr/local/lib/liblcm.so")
target_link_libraries(lcm_service ${LCM_LIBRARIES})
target_link_libraries(lcm_publisher ${LCM_LIBRARIES})
target_link_libraries(udp_publisher ${LCM_LIBRARIES})

target_link_libraries(lcm_service ${TORCH_LIBRARIES})
target_link_libraries(lcm_publisher ${TORCH_LIBRARIES})
target_link_libraries(udp_publisher ${TORCH_LIBRARIES})
target_link_libraries(udp_publisher_tinker ${TORCH_LIBRARIES})
