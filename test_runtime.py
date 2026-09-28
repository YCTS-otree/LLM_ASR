import unittest
from unittest.mock import Mock
from devices import input_devices, select_device
from engines import resolve_runtime


class RuntimeTests(unittest.TestCase):
    def test_precision_device_policy(self):
        self.assertEqual(resolve_runtime('auto', 'fp16', True), 'cuda')
        self.assertEqual(resolve_runtime('auto', 'bf16', True, True), 'cuda')
        self.assertEqual(resolve_runtime('auto', 'int8', True), 'cpu')
        self.assertEqual(resolve_runtime('auto', 'fp32', False), 'cpu')
        for args in [('cpu', 'fp16', True), ('cuda', 'int8', True),
                     ('cuda', 'bf16', True, False), ('cuda', 'fp6', True)]:
            with self.assertRaises(ValueError):
                resolve_runtime(*args)
        with self.assertRaises(RuntimeError):
            resolve_runtime('cuda', 'fp32', False)

    def test_refresh_reinitializes_before_enumeration(self):
        backend = Mock()
        backend.default.device = (1, 0)
        backend.query_hostapis.return_value = [{'name': 'WASAPI'}]
        backend.query_devices.return_value = [
            {'name': 'Speaker', 'max_input_channels': 0, 'hostapi': 0},
            {'name': 'RODE', 'max_input_channels': 2, 'hostapi': 0}]
        items, default = input_devices(True, backend)
        calls = [call[0] for call in backend.mock_calls]
        self.assertLess(calls.index('_terminate'), calls.index('_initialize'))
        self.assertLess(calls.index('_initialize'), calls.index('query_devices'))
        self.assertEqual(items, [(1, 'RODE', 'WASAPI')])
        self.assertEqual(default, 1)

    def test_preserve_microphone_when_indices_change(self):
        items = [(2, 'Internal', 'MME'), (7, 'RODE', 'WASAPI'), (8, 'RODE', 'MME')]
        self.assertEqual(select_device(items, ('RODE', 'WASAPI'), 2), 1)
        self.assertEqual(select_device(items, ('Disconnected', 'MME'), 8), 2)
        self.assertEqual(select_device([], None, -1), 0)


if __name__ == '__main__':
    unittest.main()
