import os
import lit.formats
from lit.llvm import llvm_config

config.name = "NUMERIC"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.numeric_obj_root, "test")

llvm_config.use_default_substitutions()

tool_dirs = [config.numeric_tools_dir, config.llvm_tools_dir]
tools = ["numeric-opt", "FileCheck"]
llvm_config.add_tool_substitutions(tools, tool_dirs)