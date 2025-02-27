## @defgroup Analyses-Mission-Segments Segments
# RCAIDE/Analyses/Mission/Segments/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""RCAIDE Package Setup
"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from Legacy.trunk.S.Analyses.Mission.Segments.Segment     import Segment
from Legacy.trunk.S.Analyses.Mission.Segments.Simple      import Simple
from Legacy.trunk.S.Analyses.Mission.Segments.Aerodynamic import Aerodynamic

from . import Climb
from . import Conditions
from . import Cruise
from . import Descent
from . import Ground
from . import Hover
from . import Single_Point
from . import Transition