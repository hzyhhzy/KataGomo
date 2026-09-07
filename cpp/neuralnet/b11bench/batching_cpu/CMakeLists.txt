cmake_minimum_required(VERSION 3.19)
project(b11_batching_cpu_test LANGUAGES CXX)
get_filename_component(KATAGO_CPP "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
set(cpu_sources
  core/global.cpp core/bsearch.cpp core/config_parser.cpp core/datetime.cpp
  core/fancymath.cpp core/fileutils.cpp core/hash.cpp core/logger.cpp core/makedir.cpp
  core/md5.cpp core/rand.cpp core/rand_helpers.cpp core/sha2.cpp core/timer.cpp
  game/board.cpp game/boardhistory.cpp game/rules.cpp
  neuralnet/nninputs.cpp neuralnet/sgfmetadata.cpp neuralnet/modelversion.cpp
  neuralnet/nneval.cpp neuralnet/desc.cpp search/mutexpool.cpp)
list(TRANSFORM cpu_sources PREPEND "${KATAGO_CPP}/")
foreach(test_target b11_batching_cpu_test b11_batching_fake_cuda_test)
  add_executable(${test_target}
    ../batching_cpu_test.cpp ../batching_fake_backend.cpp ${cpu_sources})
  target_compile_features(${test_target} PRIVATE cxx_std_17)
  target_include_directories(${test_target} PRIVATE
    "${KATAGO_CPP}/external" "${KATAGO_CPP}/external/filesystem-1.5.8/include")
  target_link_libraries(${test_target} PRIVATE Threads::Threads ZLIB::ZLIB)
  target_compile_definitions(${test_target} PRIVATE
    NO_GIT_REVISION NO_LIBZIP NOMINMAX BYTE_ORDER=1234 LITTLE_ENDIAN=1234 BIG_ENDIAN=4321)
  if(MSVC)
    target_link_libraries(${test_target} PRIVATE ws2_32)
    target_compile_definitions(${test_target} PRIVATE
      _CRT_SECURE_NO_WARNINGS _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_options(${test_target} PRIVATE /W3 /EHsc /MP4)
    target_link_options(${test_target} PRIVATE /STACK:8388608)
  else()
    target_compile_options(${test_target} PRIVATE -Wall -Wextra)
  endif()
endforeach()
target_compile_definitions(b11_batching_fake_cuda_test PRIVATE USE_CUDA_BACKEND)
