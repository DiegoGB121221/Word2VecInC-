cmake_minimum_required(VERSION 3.10)
project(Prueba2 C)

# Especifica la ruta de vcpkg (ajusta según tu instalación)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/Diego Covarrubias/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Busca Jansson
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(jansson_DIR "C:/Users/Diego Covarrubias/vcpkg/installed/x64-windows/share/jansson")
find_package(jansson CONFIG REQUIRED)

add_executable(prueba prueba2.c)

# Enlaza Jansson
target_include_directories(prueba PRIVATE "C:/Users/Diego Covarrubias/vcpkg/installed/x64-windows/include")
target_link_libraries(prueba PRIVATE jansson::jansson)

# Después de target_link_libraries
if(WIN32)
    add_custom_command(
        TARGET prueba POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "C:/Users/Diego Covarrubias/vcpkg/installed/x64-windows/bin/jansson.dll"
            $<TARGET_FILE_DIR:prueba>
    )
endif()


#target_link_libraries(my_app PUBLIC jansson)