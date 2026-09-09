import json
import subprocess
import unittest
from unittest.mock import patch
import collector

class CollectorTests(unittest.TestCase):
    config = {'host':'example-host','root':'/example/train','evaluation_root':'/example/eval'}
    def test_auth_failure_sanitized(self):
        with patch('collector.subprocess.run',return_value=subprocess.CompletedProcess([],255,'','Permission denied SECRET')):
            r=collector.collect(self.config)
        self.assertFalse(r['ok']);self.assertIn('認証',r['status']);self.assertNotIn('SECRET',json.dumps(r))
    def test_timeout(self):
        with patch('collector.subprocess.run',side_effect=subprocess.TimeoutExpired([],18)):
            self.assertEqual(collector.collect(self.config)['status'],'接続タイムアウト')
    def test_invalid_host(self):
        with self.assertRaises(ValueError):collector.collect({**self.config,'host':'-oProxyCommand=bad'})
    def test_readonly_batchmode(self):
        with patch('collector.subprocess.run',return_value=subprocess.CompletedProcess([],0,'{"observed_at":1,"gpus":[]}','')) as run:
            self.assertTrue(collector.collect(self.config)['ok'])
        args=run.call_args.args[0]
        self.assertIn('BatchMode=yes',args);self.assertIn('StrictHostKeyChecking=yes',args)
        self.assertEqual(run.call_args.kwargs['timeout'],18)
if __name__=='__main__':unittest.main()
