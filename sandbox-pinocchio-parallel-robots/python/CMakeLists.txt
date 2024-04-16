set(${PROJECT_NAME}_PYTHON_FILES 
    __init__.py
    reorganize_models.py
)

foreach(python ${${PROJECT_NAME}_PYTHON_FILES})
  python_install_on_site(${PY_NAME} ${python})
endforeach(python ${${PROJECT_NAME}_PYTHON_FILES})