set(${PROJECT_NAME}_PYTHON_TESTS robot)

foreach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
  add_python_unit_test("py-${TEST}" "unittest/test_${TEST}.py" ".")
endforeach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
