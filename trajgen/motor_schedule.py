"""Select velocities for torque bounds on intervals that advance simulation time."""

def control_end_velocities(speeds, phases):
    # Impulses add a solver node/state but consume no control interval. The bound
    # for an integrated node must use THAT node's end state after any impulse.
    if len(speeds) != len(phases) + 1:
        raise ValueError('solver states must have one more row than solver phases')
    return [speeds[node + 1] for node, phase in enumerate(phases)
            if not phase.endswith(':impulse')]
