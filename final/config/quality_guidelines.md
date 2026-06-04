# Quality Guidelines

Edit this file freely. Add notes for any material system or stage.
The agent reads the sections that match the current tool and material system
and uses them to assess results qualitatively after the hard rules pass.

Section header format:
  ## All Systems / <tool_name>
  ## <material keyword> / <tool_name>

---

## All Systems / run_elastic_inverse

Fit error above 0.05 usually means one of:
- Measurements are inconsistent with each other
- Wrong material selected (check fiber and polymer names)
- Not enough measurements — adding G12 or nu12 significantly helps

If only E1 and E2 are provided, aspect ratio (ar) is poorly constrained.
The result may still be useful but treat ar as approximate.

a11 below 0.50 is unusual for printed composites — check if the print direction
matches the measurement direction.

matrix_modulus outside 1500-5000 MPa is worth checking against the polymer datasheet.

---

## All Systems / run_thermoelastic_inverse

f_cte1 (fiber longitudinal CTE) should be near zero or slightly negative for carbon
fibers, and small positive for glass fibers. Large positive values (> 5 ppm/K)
suggest a mismatch — verify fiber identity.

If fit error is high but measurements look correct, check that Stage 1 was run
on the same material system. A mismatch in microstructure will carry through.

---

## All Systems / run_thermal_inverse

k_f1 (fiber longitudinal conductivity) should be much higher than k_f2 for carbon
fibers (typical ratio 5-10x). If they are similar, check the CSV column order.

At least 4 temperature points spanning 50 deg C or more are recommended.
Fewer points or a narrow range make p1 and p2 poorly constrained.

---

## Carbon Fiber / run_elastic_inverse

Expect a11 > 0.65 for most FFF-printed carbon fiber composites at standard settings.
Values below 0.55 suggest poor fiber alignment — check nozzle temperature and
print speed.

matrix_modulus below 2000 MPa is low for typical structural polymers used with
carbon fiber. If this happens, verify the polymer selection.

---

## Glass Fiber / run_elastic_inverse

E-glass composites typically show lower a11 than carbon fiber (0.55-0.75 is normal)
due to shorter fiber lengths. Expect lower E1 and higher E2 than carbon systems.
