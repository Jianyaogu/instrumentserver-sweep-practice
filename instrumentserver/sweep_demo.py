# sweep_frequency.py
import numpy as np

from instrumentserver.client import Client
from labcore.measurement.record import dependent, record_as
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.storage import run_and_save_sweep

def main():
    cli = Client(host="127.0.0.1", port=5555)

    pm = cli.find_or_create_instrument("parameter_manager")

    if "qubit.frequency"   not in pm.list(): pm.add_parameter("qubit.frequency",   initial_value=5.000, unit="GHz")
    if "qubit.pipulse.len" not in pm.list(): pm.add_parameter("qubit.pipulse.len", initial_value=40,    unit="ns")
    if "qubit.pipulse.amp" not in pm.list(): pm.add_parameter("qubit.pipulse.amp", initial_value=0.5,   unit="V")


    def measure_dummy():
        f = pm.qubit.frequency()      # GHz
        y = -((f - 5.25) ** 2) + 0.1  # a little parabola as fake signal
        return float(y)

    freqs = np.linspace(5.00, 5.50, 51)
    sweep = sweep_parameter(
        pm.qubit.frequency,           # the parameter to set each step
        freqs,                         # iterable (the "pointer")
        record_as(                     # action: measure and record as 'signal'
            measure_dummy,
            dependent("signal", unit="arb")
        )
    )

    for rec in sweep:
        print(rec)

    data = run_and_save_sweep(sweep, "./data", "freq_sweep", return_data=True)
    print("Saved sweep to:", data["_location"] if "_location" in data else "see console")

if __name__ == "__main__":
    main()
