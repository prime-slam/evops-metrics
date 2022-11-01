import evops
import pytest


@pytest.fixture(scope="function")
def clean_env():
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.75
    evops.metrics.constants.UNSEGMENTED_LABEL = 0
