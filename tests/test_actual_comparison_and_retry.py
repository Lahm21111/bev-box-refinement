import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from PIL import Image

from model_tools.comparison import save_actual_comparison
from model_runtime.gemini_box_decision import call_api


class ActualComparisonAndRetryTests(unittest.TestCase):
    def test_before_is_byte_identical_to_actual_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for side,color in [('input','green'),('updated','red')]:
                folder=root/side;folder.mkdir()
                for name in ['lidar_top','lidar_front','lidar_side','camera_crop']:
                    Image.new('RGB',(64,32),color).save(folder/f'{name}.png')
                (folder/'pixel_mapping.json').write_text(json.dumps(dict(view_frame='box')))
            save_actual_comparison(root/'comparison',root/'input',root/'updated')
            for name in ['lidar_top','lidar_front','lidar_side','camera_crop']:
                self.assertEqual((root/'input'/f'{name}.png').read_bytes(),(root/'comparison/before'/f'{name}.png').read_bytes())
                self.assertEqual((root/'updated'/f'{name}.png').read_bytes(),(root/'comparison/after'/f'{name}.png').read_bytes())
            self.assertTrue((root/'comparison/before_after.png').exists())

    def test_429_honors_retry_delay_and_retries_same_payload(self):
        error=urllib.error.HTTPError('https://example.invalid',429,'limit',{},io.BytesIO(b'Please retry in 46s'))
        success=io.BytesIO(b'{"answers": {}}')
        with patch('urllib.request.urlopen',side_effect=[error,success]) as opener,patch('time.sleep') as sleep:
            self.assertEqual(call_api(dict(model='gemini-decision-1'),'test-key',120),dict(answers={}))
        sleep.assert_called_once_with(46.)
        self.assertEqual(opener.call_count,2)
        self.assertEqual(opener.call_args_list[0].args[0].data,opener.call_args_list[1].args[0].data)

    def test_other_http_errors_are_not_retried(self):
        error=urllib.error.HTTPError('https://example.invalid',400,'invalid',{},io.BytesIO(b'bad request'))
        with patch('urllib.request.urlopen',side_effect=error),patch('time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'HTTP 400'):
                call_api({},'test-key',120)
        sleep.assert_not_called()

    def test_rate_limit_retries_are_bounded(self):
        errors=[urllib.error.HTTPError('https://example.invalid',429,'limit',{'Retry-After':'0'},io.BytesIO(b'limit')) for _ in range(2)]
        with patch('urllib.request.urlopen',side_effect=errors) as opener,patch('time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'HTTP 429'):
                call_api({},'test-key',120,max_retries=1)
        self.assertEqual(opener.call_count,2)
        sleep.assert_called_once_with(0.)
