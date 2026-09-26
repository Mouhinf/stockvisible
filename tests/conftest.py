import pytest

from stockvisible import data, splits


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "touches_test_period: lit des lignes de la période test réservée (jours 76–90 de train). "
        "Exclure avec -m 'not touches_test_period' pour un run dev-only.",
    )


@pytest.fixture(scope="module")
def dev():
    """Données RÉELLES dev (train + validation) chargées sous garde-fous : toute tentative de
    construire le test scellé ou de relire les 90 jours complets fait échouer le test."""
    if not (data.RAW_DIR / "train.parquet").exists():
        pytest.skip("lancer `python -m stockvisible.data`")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("période test touchée")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(splits, "chronological_split", forbidden)
        mp.setattr(splits.SealedFrame, "__init__", forbidden)
        mp.setattr(data, "load_raw", forbidden)
        yield splits.load_dev()
