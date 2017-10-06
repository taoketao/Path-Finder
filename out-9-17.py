# IPython log file


from cleaner import *
from gym import Env
np.random.seed(7)

e = Env()
def _new_reset():
    return ex.get_random_starting_state()['state']
e._reset = _new_reset
e.CURRENT_STATE_MRB = e.reset()

e._render = print_state(e.CURRENT_STATE_MRB, 'condensed', 'string_ret')
