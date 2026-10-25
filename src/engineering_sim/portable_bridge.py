"""Bridge used only in a deliberately exported flat scientific package."""
from pathlib import Path
from unified_sim_engine import operation,report_failure
import unified_sim_engine
def source_files():return [Path(__file__).resolve(),Path(unified_sim_engine.__file__).resolve()]
