"""P0 smoke tests: the package imports and is installed with its declared version."""

from importlib.metadata import version

import podrank


def test_package_imports() -> None:
    assert podrank.N_SEATS == 4


def test_public_exports_are_declared() -> None:
    assert podrank.__all__ == sorted(set(podrank.__all__))
    for name in podrank.__all__:
        assert hasattr(podrank, name)


def test_installed_version_matches_source() -> None:
    assert version("podrank") == podrank.__version__
