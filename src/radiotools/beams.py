# -*- coding: utf-8 -*-
"""Instrument beams definitions.

This module creates the classed and method needed for the definition of a instrument as a **radiotelescope**.

Example:
    Two instruments are predefined as instances of this class. **Uirapuru** and **Callisto**.

Todo:
    * Tudo
    * Depois de tudo

"""

class beams:
    def __init__(self):
        self.name = None
        self.NPOINTS = None
        self.DIM = None
        self.response = None
        self.view = None
        return None

    def get_FWHM(self):
        return None

    def get_Directivity(self):
        return None

    def get_Efficiency(self):
        return None

    def get_Area_eff(self):
        return None

    def get_Area_geom(self):
        return None

    def get_Taper_angle(self):
        return None

    def get_TaperDB(self):
        return None

    def get_Omega_beam(self):
        return None

    def get_Omega_main(self):
        return None

    def set_Response(self):
        return None

    def set_View(self):
        return None

    def calculate(self):
        return None
