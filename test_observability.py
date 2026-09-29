import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from observability import configure_logging


class LoggingTests(unittest.TestCase):
    def test_console_and_file_are_both_live_without_duplicate_handlers(self):
        logger=logging.getLogger('meeting_asr')
        previous=logger.handlers[:];level=logger.level;propagate=logger.propagate
        logger.handlers=[]
        try:
            with tempfile.TemporaryDirectory() as folder:
                output=io.StringIO()
                with patch('observability.sys.stderr',output),patch.dict('os.environ',{'ASR_DEBUG_LLM':'1'}):
                    configure_logging(folder);configure_logging(folder)
                    logger.info('visible operational event')
                    logger.debug('file-only debug event')
                self.assertEqual(output.getvalue().count('visible operational event'),1)
                self.assertNotIn('file-only debug event',output.getvalue())
                text=Path(folder,'pipeline.log').read_text(encoding='utf-8')
                self.assertEqual(text.count('visible operational event'),1)
                self.assertIn('file-only debug event',text)
                for handler in logger.handlers:handler.close()
        finally:
            logger.handlers=previous;logger.setLevel(level);logger.propagate=propagate
