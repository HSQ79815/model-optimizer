#cmake_policy(SET CMP0104 OLD)

# name
set(RED_PLUGINS "red_plugins")

# src file
file(GLOB PLUGIN_SOURCES *.cc)
file(GLOB PLUGIN_CU_SOURCES *.cu)

set(PLUGIN_LISTS
    hardswish
    custom_div
)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

add_library(${RED_PLUGINS} SHARED ${PLUGIN_SOURCES})
