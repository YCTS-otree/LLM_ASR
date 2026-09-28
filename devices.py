"""Refresh PortAudio's enumeration snapshot only when no stream is active."""
def input_devices(refresh=False, backend=None):
    if backend is None:
        import sounddevice as backend
    if refresh:
        backend._terminate()
        backend._initialize()
    hosts = backend.query_hostapis()
    default = backend.default.device[0]
    items = []
    for index, info in enumerate(backend.query_devices()):
        if info['max_input_channels'] > 0:
            host = hosts[info['hostapi']]['name']
            items.append((index, info['name'], host))
    return items, default


def select_device(items, previous, default):
    # Numeric indices can change after hot-plug; preserve name and host API.
    for row, (_, name, host) in enumerate(items):
        if (name, host) == previous:
            return row
    return next((row for row, (index, _, _) in enumerate(items) if index == default), 0)
