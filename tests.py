"""An extremely ad-hoc test harness for basic sanity checking.
Make sure things don't revert /too/ badly when I commit things.

To add a test, just add a new function annotated with the decorator `@test`.
"""

import traceback
from termcolor import colored

import torch

from ref_model import ConvBlock


### DEFINITIONS ### 
test_registry = {}

def test(fn):
    test_registry[fn.__name__] = fn
    return fn



### TESTS ###
@test
def _test_conv_block():
    """Just feed some numbers through a ConvBlock and make sure it doesn't
    crash.
    """
    input_shape = (10, 20)
    input_tensor = torch.randn(1, 1, *input_shape)  # need these leading 1s for batch size and...?
    cb = ConvBlock(input_shape, 3, 2, [32, 32, 64])
    cb(input_tensor)



### RUN THE TESTS ###
for name, fn in test_registry.items():
    try:
        fn()
        print(colored("Passed test {}".format(name), 'green'))
    except Exception :  # Here, pokemon exception handling is a-ok
        print(colored("Tests failed in the execution of {}\n\n".format(name), 'red'),
              traceback.format_exc())
        exit(1)
