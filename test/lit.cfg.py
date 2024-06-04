import lit.formats
import os

config.name = 'ParMGMC'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cc']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root)

config.substitutions.append(('%cxx', config.parmgmc_cxx))
config.substitutions.append(('%flags', config.parmgmc_comp))
config.substitutions.append(('%mpirun', config.parmgmc_mpirun))
