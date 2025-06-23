import Code_for_Experiment.RAG.clap_finetuning_zh_en.random_cut_mp3
import numpy
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")

@pytest.fixture
def random():
    numpy.random.seed(42)
    return numpy.random