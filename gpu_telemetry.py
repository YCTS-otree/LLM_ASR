"""Fixed-interval nvidia-smi power/VRAM sampling; no additional GPU SDK."""
import subprocess
import threading
import time
import statistics


class GPUSampler:
    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []
        self.process = None
        self.error = None

    def start(self):
        try:
            self.process = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=memory.used,power.draw,utilization.gpu',
                 '--format=csv,noheader,nounits', f'--loop-ms={self.interval_ms}', '-i', '0'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except OSError as exc:
            self.error = str(exc)
        return self

    def _read(self):
        for line in self.process.stdout:
            try:
                memory, power, utilization = [float(x.strip()) for x in line.split(',')]
                self.samples.append(dict(time=time.perf_counter(), memory_mib=memory,
                                         power_w=power, utilization=utilization))
            except ValueError:
                self.error = 'Some nvidia-smi measurements were unavailable'

    def summarize(self, start, end):
        samples = [s for s in self.samples if start <= s['time'] <= end]
        if not samples:
            return dict(samples=0, average_power_w=None, peak_power_w=None,
                        peak_device_memory_mib=None, approximate_energy_j=None)
        energy = sum((a['power_w'] + b['power_w']) / 2 * (b['time'] - a['time'])
                     for a, b in zip(samples, samples[1:]))
        # Boundary rectangles cover the two partial sample intervals.
        energy += samples[0]['power_w'] * (samples[0]['time'] - start)
        energy += samples[-1]['power_w'] * (end - samples[-1]['time'])
        return dict(samples=len(samples), average_power_w=statistics.mean(s['power_w'] for s in samples),
                    peak_power_w=max(s['power_w'] for s in samples),
                    peak_device_memory_mib=max(s['memory_mib'] for s in samples),
                    approximate_energy_j=energy)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.thread.join(timeout=2)
            self.process.stdout.close()
