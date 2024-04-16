set(${PROJECT_NAME}_PYTHON_TESTS load)

foreach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
  add_python_unit_test("py-${TEST}" "unittest/test_${TEST}.py" python)
endforeach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})