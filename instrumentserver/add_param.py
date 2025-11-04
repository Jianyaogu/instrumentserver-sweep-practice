from instrumentserver.client import Client

cli = Client(host="127.0.0.1", port=5555)
pm = cli.find_or_create_instrument('parameter_manager')

# Add three parameters under a 'qubit' group
pm.add_parameter("qubit.frequency",    initial_value=5.000, unit="GHz")
pm.add_parameter("qubit.pipulse.len",  initial_value=40,    unit="ns")
pm.add_parameter("qubit.pipulse.amp",  initial_value=0.5,   unit="V")

# Check they exist
print("Parameters:", pm.list())
print("frequency ->", pm.qubit.frequency())
