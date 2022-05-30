#cmake_policy(SET CMP0104 OLD)

# name
set(ORT_PLUGINS "ort_custom_op_plugins")

# src file
file(GLOB PLUGIN_SOURCES *.cc)
file(GLOB PLUGIN_CU_SOURCES *.cu)

set(PLUGIN_LISTS
    "custom_div"
)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

if(BUILD_SHARED_LIBS)
    add_library(${ORT_PLUGINS} SHARED ${PLUGIN_SOURCES})
else()
    add_library(${ORT_PLUGINS} STATIC ${PLUGIN_SOURCES})
endif()

target_link_libraries(${ORT_PLUGINS} 
	PUBLIC ${EXTRALIBS}
)

