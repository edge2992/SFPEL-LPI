import pytest
import scipy.io as spio


@pytest.fixture
def data():
    from src.loader import InputData

    return InputData()


@pytest.fixture
def LNS_LNCRNA():
    mat = spio.loadmat("tests/data/LNS_LNCRNA_twister1337.mat", squeeze_me=True)
    return mat["result"]


def test_LNC_calculate(data, LNS_LNCRNA):
    from src.featurization import fast_LNC_calculate

    input = data.SCPseDNCFeature_LNCRNA
    result = fast_LNC_calculate(input, input.shape[0])
    assert result.shape == LNS_LNCRNA.shape
    assert (abs(result - LNS_LNCRNA) < 0.0001).all()
