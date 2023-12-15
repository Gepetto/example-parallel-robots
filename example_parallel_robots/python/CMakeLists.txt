set(${PROJECT_NAME}_PYTHON_FILES 
robot_utils.py 
actuation_model.py 
closed_loop_mount.py 
constraints.py 
loader_tools.py 
robot_options.py
talos_closed.py
__init__.py 
__main__.py 
)

foreach(python ${${PROJECT_NAME}_PYTHON_FILES})
  python_install_on_site(${PY_NAME} ${python})
endforeach(python ${${PROJECT_NAME}_PYTHON_FILES})

message(${PY_NAME})
configure_file(${PY_NAME}/path.py.in ${PY_NAME}/path.py)
message(================== COPIED FILE TO ${PY_NAME}/path.py)
# python_install_on_site(${PY_NAME} path.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/path.py
        DESTINATION "${PYTHON_SITELIB}/${PY_NAME}")