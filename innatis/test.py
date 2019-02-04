import pytest

# The config is set to not run slow tests
# some tests take ~30s
# because loading a model into memory the first time is slow

test_config = ["-m",
               "not slow",
               "--verbose",
               "--color=yes"]
# To run all tests, including slow ones, use
# test_config = ["--verbose", "--pep8", "--color=yes"]

pytest.main(test_config)
