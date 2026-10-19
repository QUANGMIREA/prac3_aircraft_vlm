import tempfile
import unittest
from pathlib import Path

from main import parse_prediction, load_dataset
from evaluate import metrics


class AircraftTask3Tests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_prediction('MILITARY'), 'MILITARY')
        self.assertEqual(parse_prediction('civilian.'), 'CIVILIAN')
        self.assertEqual(parse_prediction('I think it is a military jet'), 'INVALID')

    def test_metrics(self):
        x = metrics([
            ('MILITARY', 'MILITARY'),
            ('MILITARY', 'CIVILIAN'),
            ('CIVILIAN', 'CIVILIAN'),
            ('CIVILIAN', 'MILITARY'),
        ])
        self.assertEqual((x['tp'], x['fp'], x['tn'], x['fn']), (1, 1, 1, 1))
        self.assertEqual(x['accuracy'], 0.5)
        self.assertEqual(x['f1'], 0.5)

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'dataset.csv'
            p.write_text('image,label,difficulty,source_url\na.jpg,MILITARY,easy,https://example.com\n',
                         encoding='utf-8')
            self.assertEqual(len(load_dataset(p)), 1)


if __name__ == '__main__':
    unittest.main()
