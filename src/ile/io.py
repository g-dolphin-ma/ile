import logging
from pathlib import Path
import pandas as pd
import requests
import io
import zipfile

__all__ = ['NHAMCS']

logger = logging.getLogger(__name__)

class NHAMCS:
    """
    A wrapper class that represents NHAMCS dataset from CDC.
    
    Args:
        year (int): the year to dowload
    """

    DEFAULT_DATA_DIR = Path('data')
    CDC_BASE_URL = r'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/'

    def __init__(self, year: int) -> None:
        self._filename = self.DEFAULT_DATA_DIR / f'ed{year}-stata.dta'
        if not self._filename.exists():
            self._download_from_cdc()
        
        self._reader = pd.read_stata(self._filename, iterator=True)
        self._data = self._reader.read(convert_categoricals=False)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def variable_labels(self):
        return self._reader.variable_labels()

    @property
    def data_label(self):
        return self._reader.data_label

    @property
    def value_labels(self):
        return self._reader.value_labels()

    def _download_from_cdc(self) -> None:
        """Download the DTA file from CDC."""

        try:
            # 1) download
            download_name = self._filename.with_suffix('.zip').name
            download_url = self.CDC_BASE_URL + download_name
            r = requests.get(download_url)
            r.raise_for_status()

            # 2) extract
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self._filename.parent)
        except requests.exceptions.HTTPError:
            raise requests.exceptions.HTTPError(f"unable to find '{download_name}' on CDC")
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(f"'{download_name}' is malformed ZIP file")