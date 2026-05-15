"""
paleo_constants.py
==================
All physics and biomechanical constants for the Ankylosaurus RL simulation.

Primary sources:
  [ARBOUR2009]  Arbour & Snively 2009. "Finite element analyses of ankylosaurid
                dinosaur tail club impacts." Anatomical Record 292(9):1412-1426.
                PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2726940/
  [ARBOUR2013]  Arbour & Currie 2013. "Euoplocephalus tutus and the diversity
                of ankylosaurid dinosaurs." PLoS ONE 8(5):e62421.
  [HUTCHINSON]  Hutchinson & Garcia 2002. "Tyrannosaurus was not a fast runner."
                Nature 415:1018-1021.
  [ALEXANDER]   Alexander 1989. "Dynamics of Dinosaurs and Other Extinct Giants."
                Columbia University Press.
  [HELLCREEK]   Fastovsky & Sheehan 2005. "The Extinction of the Dinosaurs in
                North America." GSA Today 15(4):4-10.
"""

# ============================================================
# BODY MASS
# ============================================================

BODY_MASS_KG = 5000.0
"""
Approximate body mass in kilograms.
Range: 4,800–6,000 kg [ARBOUR2013].
We use 5,000 kg as central estimate.
MuJoCo model total mass should be verified to sum near this value.
"""

BODY_MASS_KG_MIN = 4800.0
BODY_MASS_KG_MAX = 6000.0

# ============================================================
# LOCOMOTION
# ============================================================

MAX_SPEED_MS = 3.6
"""
Maximum locomotion speed in m/s (~13 km/h).
Derived from:
  - Limb bone stress limits (scaling from extant large quadrupeds) [ALEXANDER]
  - Wide-gauge trackway stride length analysis
  - Upper bound; typical preferred speed much lower (~1.4 m/s)
Range: 2.2–3.6 m/s (8–13 km/h)
"""

PREFERRED_SPEED_MS = 1.4
"""
Preferred / energetically optimal speed in m/s (~5 km/h).
Derived from body-mass cost-of-transport (CoT) scaling laws.
Similar to extant large reptiles at equivalent mass. # ASSUMPTION (soft tissue)
"""

HIP_HEIGHT_M = 1.7
"""
Hip height above ground in meters.
Estimated from limb bone measurements (femur + tibia length + foot).
[ARBOUR2013] skeletal reconstructions.
"""

CENTER_OF_MASS_HEIGHT_FRACTION = 0.40
"""
CoM height as fraction of total body height.
Very low due to heavy osteoderm armor on dorsal surface.
Inferred from osteoderm mass distribution [ARBOUR2013]. # ASSUMPTION (soft tissue)
"""

GAIT_TYPE = "wide_gauge_quadruped"
"""
Wide-gauge quadruped gait.
Derived from:
  - Hip socket (acetabulum) orientation in fossil — faces laterally
  - Limb bone proportions: short, robust, positioned wide
  - Trackway evidence where available
"""

# ============================================================
# TAIL CLUB BIOMECHANICS
# ============================================================

TAIL_CLUB_IMPACT_FORCE_N_MIN = 7281.0
TAIL_CLUB_IMPACT_FORCE_N_MAX = 14360.0
"""
Tail club impact force range in Newtons.
Source: [ARBOUR2009] FEA using CT-scan-derived 3D models of club knobs.
Table 4: 7,281 N (small knob) to 14,360 N (large knob) at peak impact.
This is the PRIMARY FEA-grounded parameter in the simulation.
"""

TAIL_CLUB_IMPACT_STRESS_MPA_MIN = 364.0
TAIL_CLUB_IMPACT_STRESS_MPA_MAX = 718.0
"""
Von Mises stress at impact site on target bone (MPa).
Source: [ARBOUR2009] Table 4.
Cortical bone yield stress ~150–200 MPa → club exceeds fracture threshold.
This grounds the damage model: a full-force strike should stagger a T-rex.
"""

TAIL_HANDLE_DOF = "lateral_only"
"""
The tail handle (distal caudal vertebrae) is ossified and interlocked.
This restricts motion to lateral swing ONLY — no dorsoventral flex.
Source: [ARBOUR2009] — caudal vertebra morphology, neural spine fusion.
Implemented in MJCF as: handle joint has only tail_handle_lat, NO dv joint.
"""

TAIL_CLUB_MASS_KG = 30.0
"""
Approximate mass of the tail knob (club) in kg.
Estimated from knob volume (CT scan) × bone density (~1.9 g/cm³).
Large knob estimate. [ARBOUR2009]. # ASSUMPTION (density estimate)
"""

TAIL_CLUB_RADIUS_M = 0.22
"""
Approximate radius of the club knob sphere in meters.
Based on large Ankylosaurus knob dimensions. [ARBOUR2013].
"""

# ============================================================
# ARMOR (OSTEODERMS)
# ============================================================

OSTEODERM_COVERAGE = "dorsal_and_lateral"
"""
Osteoderms cover the dorsal and lateral surfaces of the body, neck, and tail.
Does NOT cover ventral surface — this is the behavioral vulnerability.
Grounded in fossil skin impressions and articulated specimens.
"""

OSTEODERM_TOTAL_MASS_FRACTION = 0.08
"""
Estimated fraction of total body mass in osteoderms: ~8% (~400 kg).
# ASSUMPTION — no direct measurement available for Ankylosaurus specifically.
Inferred from related ankylosaurs and extant armored analogues (crocodilians).
"""

T_REX_BITE_FORCE_N = 57000.0
"""
T-rex maximum bite force in Newtons (~57 kN).
Source: Bates & Falkingham 2012, Gignac & Erickson 2017.
Osteoderm armor modeled as absorbing a fraction of this before health depletes.
"""

ARMOR_BITE_ABSORPTION_FRACTION = 0.85
"""
Fraction of T-rex bite force absorbed by osteoderm armor on dorsal surface.
# ASSUMPTION — no direct measurement. Grounded in:
  - Osteoderms observed with bite marks but intact in fossil record
  - Suggests armor was functionally effective as bite deflection
"""

# ============================================================
# HELL CREEK ENVIRONMENT
# ============================================================

AMBIENT_TEMP_C = 25.0
"""
Approximate ambient temperature in Hell Creek Formation (~66 Ma).
Warm subtropical / sub-humid climate. [HELLCREEK]
Not directly used in physics but informs activity level assumptions.
"""

VEGETATION_TYPES = ["cycad", "fern", "conifer", "angiosperm"]
"""
Primary vegetation types in Hell Creek Formation.
Angiosperms (flowering plants) were diversifying rapidly at this time.
[HELLCREEK] — Fastovsky & Sheehan 2005.
Modeled as food source geoms in the environment.
"""

FOOD_ENERGY_KCAL_PER_KG_VEGETATION = 1800.0
"""
Approximate caloric content of vegetation in kcal/kg dry mass.
# ASSUMPTION — uses modern analogue plant caloric values.
Scaled to ankylosaur daily energy requirements (~50,000 kcal/day for 5t animal).
"""

DAILY_ENERGY_REQUIREMENT_KCAL = 50000.0
"""
Estimated daily caloric requirement.
# ASSUMPTION — derived from metabolic scaling:
  BMR ~ 70 × (mass_kg)^0.75 kcal/day for endotherm-like metabolism
  = 70 × 5000^0.75 ≈ 70 × 707 ≈ 49,500 kcal/day
Actual metabolism unknown; mesothermy possible for large dinosaurs.
"""

# ============================================================
# PREDATOR (T-REX) PARAMETERS
# ============================================================

TREX_MAX_SPEED_MS = 5.5
"""
T-rex maximum speed in m/s (~20 km/h).
Source: [HUTCHINSON] — biomechanical upper bound.
Pretrained mesozoic-labs T-rex achieves ~3.47 m/s avg forward velocity
at Stage 2 (locomotion stage). Use 5.5 m/s as absolute cap in env.
"""

TREX_DETECTION_RANGE_M = 15.0
"""
Distance at which T-rex detects and begins pursuing prey.
# ASSUMPTION — based on estimated visual acuity for large theropods
and comparison to extant large predators.
"""

TREX_BITE_RANGE_M = 1.5
"""
Distance at which T-rex bite attack can register.
Approximated from skull length + neck reach. # ASSUMPTION
"""

TREX_ATTACK_COOLDOWN_S = 3.0
"""
Minimum time between bite attempts.
# ASSUMPTION — based on jaw reset mechanics of large theropods.
"""

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

SIM_TIMESTEP_S = 0.005
"""MuJoCo timestep in seconds (200 Hz). Standard for contact-rich locomotion."""

CONTROL_TIMESTEP_S = 0.02
"""Policy control timestep in seconds (50 Hz). 4 sim steps per control step."""

EPISODE_MAX_STEPS = 15000
"""
Max steps per episode at control frequency.
15,000 × 0.02s = 300s = 5 minutes of simulated survival.
"""

FALL_HEIGHT_THRESHOLD_M = 0.6
"""
If torso z-position drops below this, episode ends (fallen).
0.6m ≈ 35% of hip height → effectively on the ground.
"""
