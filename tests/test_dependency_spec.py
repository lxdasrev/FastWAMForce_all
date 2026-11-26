from pathlib import Path

from packaging.specifiers import SpecifierSet


def test_torch_public_version_pin_accepts_cuda_local_builds():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"torch==2.7.1"' in pyproject
    assert '"torchvision==0.22.1"' in pyproject
    assert SpecifierSet("==2.7.1").contains("2.7.1+cu126")
    assert SpecifierSet("==2.7.1").contains("2.7.1+cu128")
    assert not SpecifierSet("==2.7.1").contains("2.8.0")
