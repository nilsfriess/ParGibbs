import lit.formats
import os

config.name = 'ParMGMC'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.c']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root)

config.substitutions.append(('%cc', config.parmgmc_cc))
config.substitutions.append(('%flags', config.parmgmc_comp))
config.substitutions.append(('%mpirun', config.parmgmc_mpirun))

try:
    NP = lit_config.params['NP']
except KeyError as e:
    NP = 1
config.substitutions.append(('%NP', NP))

try:
    opts = lit_config.params['opts']
except KeyError as e:
    opts = ""

config.substitutions.append(('%opts', opts))
