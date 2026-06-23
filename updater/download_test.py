import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd

from download import Ons


def make_ons(frequency):
    return Ons(
        download_path="/tmp",
        title="Test ONS Series",
        url="https://api.beta.ons.gov.uk/v1/data?uri=/test",
        frequency=frequency,
        tags=["United Kingdom"],
    )


def mock_response(json_data):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


class OnsDownloadTest(unittest.TestCase):
    @patch("download.requests.get")
    def test_downloads_beta_monthly_series(self, get):
        get.return_value = mock_response(
            {
                "months": [
                    {"date": "2026 APR", "value": "3.0"},
                    {"date": "2026 MAY", "value": "3.1"},
                ]
            }
        )

        df = make_ons("MS").download()

        self.assertEqual(df.index.name, "date")
        self.assertEqual(list(df.columns), ["value"])
        self.assertEqual(df.loc[pd.Timestamp("2026-05-01"), "value"], 3.1)

    @patch("download.requests.get")
    def test_downloads_beta_quarterly_series(self, get):
        get.return_value = mock_response(
            {
                "quarters": [
                    {"date": "2025 Q3", "value": "0.2"},
                    {"date": "2025 Q4", "value": "0.1"},
                ]
            }
        )

        df = make_ons("Q").download()

        self.assertEqual(df.index.name, "date")
        self.assertEqual(df.loc[pd.Timestamp("2025-10-01"), "value"], 0.1)

    @patch("download.requests.get")
    def test_raises_for_missing_frequency_bucket(self, get):
        get.return_value = mock_response({"months": []})

        with self.assertRaisesRegex(ValueError, "No ONS months observations"):
            make_ons("MS").download()

    @patch("download.requests.get")
    def test_drops_non_numeric_values(self, get):
        get.return_value = mock_response(
            {
                "months": [
                    {"date": "2026 APR", "value": ""},
                    {"date": "2026 MAY", "value": "3.1"},
                ]
            }
        )

        df = make_ons("MS").download()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.index[0], pd.Timestamp("2026-05-01"))


if __name__ == "__main__":
    unittest.main()
