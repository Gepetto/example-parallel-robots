set(${PROJECT_NAME}_PYTHON_FILES
__init__.py 
actuation.py
jacobian.py
mounting.py 
constraints.py 
forward_kinematics.py
inverse_kinematics.py 
)

foreach(python ${${PROJECT_NAME}_PYTHON_FILES})
  python_install_on_site(${PY_NAME} ${python})
endforeach(python ${${PROJECT_NAME}_PYTHON_FILES})