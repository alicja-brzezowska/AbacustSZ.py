"""Light-cone shells and LRG snapshots of the ACT x DESI forward model."""

STEPS = [
    "Step0571-0576",
    "Step0582-0587",
    "Step0588-0593",
    "Step0594-0598",
    "Step0599-0604",
    "Step0605-0610",
    "Step0611-0616",
    "Step0617-0622",
    "Step0623-0627",
    "Step0628-0634",
    "Step0635-0640",
    "Step0641-0646",
    "Step0647-0652",
    "Step0653-0658",
    "Step0659-0664",
    "Step0665-0670",
    "Step0671-0676",
    "Step0677-0682",
    "Step0683-0688",
    "Step0689-0695",
]

LRG_REDSHIFTS = ["0.503", "0.542", "0.582", "0.625", "0.671", "0.718", "0.769"]


def healcounts_file(step):
    return f"LightCone0_total_heal-counts_{step}.asdf"
