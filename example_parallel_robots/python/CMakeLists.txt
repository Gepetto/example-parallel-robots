set(${PROJECT_NAME}_PYTHON_FILES 
actuation_model.py 
loader_tools.py 
robot_options.py
talos_closed.py
freeze_joints.py
__init__.py 
__main__.py 
)

foreach(python ${${PROJECT_NAME}_PYTHON_FILES})
  python_install_on_site(${PY_NAME} ${python})
endforeach(python ${${PROJECT_NAME}_PYTHON_FILES})

configure_file(${PY_NAME}/path.py.in ${PY_NAME}/path.py)
message(================== COPIED FILE TO ${PY_NAME}/path.py)
# python_install_on_site(${PY_NAME} path.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/path.py
        DESTINATION "${PYTHON_SITELIB}/${PY_NAME}")