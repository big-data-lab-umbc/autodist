# coding: utf-8

import pathlib

#import tools

# ---------------------------------------------------------------------------- #
# Base gradient aggregation rule class

class _GAR:
  """ Base gradient aggregation rule class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    """ Unimplemented constructor, no graph available at this time.
    Args:
      nbworkers Total number of workers
      nbbyzwrks Declared number of Byzantine workers
      args      Command line argument list
    """
    raise NotImplementedError

  def aggregate(self, gradients):
    """ Build the gradient aggregation operation of the given gradients.
    Args:
      gradients Computed gradient tensors
    Returns:
      Aggregated gradient tensor
    """
    raise NotImplementedError

# ---------------------------------------------------------------------------- #
# GAR script loader

# Register instance
#_register   = tools.ClassRegister("GAR")
#register    = _register.register
#instantiate = _register.instantiate
#del _register

# Load all local modules
for path in pathlib.Path(__file__).parent.iterdir():
  if path.is_file() and path.suffix == ".py" and path.stem != "__init__":
    try:
      __import__(__package__, globals(), locals(), [path.stem], 0)
    except:
      head = "[RUNNER] (" + path.stem + ") "
     # with tools.TerminalColor("warning"):
      #  import traceback
      #  for line in traceback.format_exc().splitlines():
      #    print(head + line)
      #  print("[RUNNER] Loading failed for experiment module " + repr(path.name))
