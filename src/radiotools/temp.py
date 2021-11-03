# -*- coding: utf-8 -*-
"""temporary funcion definitions."""

def get_star_cone(self, objects = None):
    # --------------------
    # Generate positions in dataframe.
    # --------------------
    if self.instrument is not None:
        timevector = self.timevector
        fwhm = self.instrument.fwhm
        observer = self._earth + self.instrument.observatory
        if objects is None:
            objects = self.local_objects
        object_list = []
        for index, star in objects.iterrows():
            ra = (star.RA*u.deg).to(u.hourangle).value
            dec = star.DEC
            celestial = api.Star(ra_hours=ra, dec_degrees=dec)
            pos = observer.at(trial_run.timevector).observe(celestial).apparent()
            ra, dec, dist = pos.radec()
            cone = observer.at(timevector).from_altaz(alt_degrees=self.instrument.Alt, az_degrees=self.instrument.Az).separation_from(pos)
            df = pd.DataFrame(zip(timevector.tai, ra._degrees, dec.degrees, cone.degrees, dist.km),
                              columns=['TIME', 'RA', 'DEC', 'ANGLE', 'DISTANCE'])
            df['NAME'] = [star.NAME] * len(timevector)
            object_list.append(df)
        objects_df = pd.concat(object_list)
        df = objects_df[objects_df.ANGLE < fwhm]
    else:
        print("Instrument not set")
        df = None
    return df
