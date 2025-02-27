## @ingroup Analyses-Mission-Segments-Descent
# Constant_CAS_Constant_Rate.py
#
# Created:  
# Modified: Aug 2020, S. Karpuk

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from Legacy.trunk.S.Methods.Missions import Segments as Methods

from Legacy.trunk.S.Analyses.Mission.Segments.Climb.Unknown_Throttle import Unknown_Throttle

# Units
from Legacy.trunk.S.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Descent
class Constant_CAS_Constant_Rate(Unknown_Throttle):
    
    def __defaults__(self):
        """ This sets the default solver flow. Anything in here can be modified after initializing a segment.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """            
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start      = None # Optional
        self.altitude_end        = 10. * Units.km
        self.descent_rate        = 3.  * Units.m / Units.s
        self.calibrated_airspeed = 100 * Units.m / Units.s
        self.true_course         = 0.0 * Units.degrees 
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        # only need to change one setup step from constant_speed_constant_rate
        initialize = self.process.initialize
        initialize.conditions = Methods.Descent.Constant_CAS_Constant_Rate.initialize_conditions
       
        return

