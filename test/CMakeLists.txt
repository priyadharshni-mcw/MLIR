configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NUMERIC_TEST_DEPENDS
  FileCheck
  numeric-opt
)

add_lit_testsuite(check-numeric "Running the numeric dialect regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NUMERIC_TEST_DEPENDS}
)
set_target_properties(check-numeric PROPERTIES FOLDER "Tests")