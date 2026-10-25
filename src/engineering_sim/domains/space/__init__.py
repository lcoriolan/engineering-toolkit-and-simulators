#   density.AirColumn   ONE air column: core.env below core_top -> US76 -> Vallado exponential, blended at 86 km
#   thermosphere.Thermosphere  NRLMSIS 2.1/2.0/00 density (F10.7, Ap, local time, lat) above 84 km, 3-D GPU table
#   forces              J2-J4, Sun/Moon, SRP, drag (host float64 reference)
#   propagate.OrbitBatch  batched GPU (Warp float64) RK4 propagation, same force terms
#   ascent.fly          ground-to-orbit point-mass launcher on the same air column
#   attitude.simulate   rigid-body attitude: gravity gradient, tilted dipole, magnetorquers, reaction wheels
#   tle                 SGP4 initial conditions / cross-check;  frames, ephem, constants
