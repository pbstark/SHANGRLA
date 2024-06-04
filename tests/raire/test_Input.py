import pytest
import sys

from shangrla.raire.raire_utils import load_contests_from_raire

class TestInput:
  
  def test_load_raire_file(self):
    assert len(load_contests_from_raire("./tests/raire/data/Aspen_2009_Mayor.raire")) == 2

if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
