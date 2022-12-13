import copy
from pprint import pprint

"""
All user parameters are listed here, plus some handy function(s).
"""

_USER_PARAMETERS = { 

    "problem": { 
        # gamma1 is the step decrease factor. Another related parameter - step
        # increase factor - is obtained as a reciprocal: gamma2 = 1 / gamma1:
        "gamma1": 0.5,

        # gamma2
        "gamma2": 1.5,

        # Residual tolerance:
        "res_tol":  1e-4, 

        # Initial step size (delta t) in the inner loop:
        "ini_stepsize": 0.01,

        # Initial time
        "initial_time": 0,

        # Final time
        "final_time": 1, 
    }, 
}

def getParameters(print_par: bool=True) -> dict:

    """
    Returns a deep copy of user parameters modified by those specified
    as command-line options.
    """ 

    assert isinstance(_USER_PARAMETERS, dict)
    user_params = copy.deepcopy(_USER_PARAMETERS)

    if print_par:
        print("-" * 80)
        print("User parameters:")
        pprint(user_params)
        print("-" * 80)
    return user_params

