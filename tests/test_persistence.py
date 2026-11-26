"""Posterior archive persistence round-trip and validation (C.2.4)."""

from __future__ import annotations

import dataclasses
import json
import warnings
import zipfile
from types import MappingProxyType

import numpy as np
import pytest

from podrank import Field, FieldContext, GameTable, Posterior, fit
from podrank.errors import PersistenceError
from podrank.rating import _archive
from podrank.rating._archive import read_posterior_archive, write_posterior_archive
from podrank.rating.design import Pattern, PatternTable, design_row
from tests._fixtures import make_game, registries
from tests.test_rating_fit import _f11_rows


def _posterior() -> Posterior:
    agents, decks = registries()
    row = make_game(
        "g1",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        winner_seat=0,
    )
    return fit(GameTable.from_records([row]), agents, decks)


def test_posterior_round_trip(tmp_path) -> None:
    post = _posterior()
    path = tmp_path / "posterior.podrank"
    post.save(path)
    loaded = Posterior.load(path)

    assert loaded.parameter_index.names == post.parameter_index.names
    assert np.allclose(loaded.mode, post.mode)
    assert np.allclose(loaded.lower_factor, post.lower_factor)
    assert np.allclose(loaded.row_space, post.row_space)
    assert np.allclose(loaded.precision, post.precision)
    assert loaded.counts() == post.counts()
    assert loaded.provenance() == post.provenance()
    assert loaded.outcome_rates() == post.outcome_rates()
    assert loaded.cluster_table == post.cluster_table
    assert loaded.cluster_strata == post.cluster_strata
    assert loaded.contrast("A", "H").status == post.contrast("A", "H").status
    assert loaded.log_worth() == post.log_worth()
    assert loaded.headline_eligibility() == post.headline_eligibility()
    assert loaded.components() == post.components()
    assert loaded.seats() == post.seats()
    assert loaded.decks() == post.decks()
    assert loaded.identification_report() == post.identification_report()
    assert loaded.predict_proba(
        [{"agents": ("H", "A", "B", "C"), "decks": ("d1", "d2", "d3", "d4")}]
    ) == post.predict_proba([{"agents": ("H", "A", "B", "C"), "decks": ("d1", "d2", "d3", "d4")}])

    field = Field.from_contexts(
        "f",
        1,
        [
            FieldContext(
                decks=("d1", "d2", "d3", "d4"),
                swap_counts=(0, 0, 0, 0),
                opponents=("H", "H", "H"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    assert loaded.sdwp(field).rows == post.sdwp(field).rows
    assert np.array_equal(
        loaded.sample(5, seed=7),
        post.sample(5, seed=7),
    )


def _seeded_cluster_weights(post: Posterior, seed: int) -> np.ndarray:
    keys = tuple(sorted(post.cluster_table))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0,))))
    return np.bincount(rng.integers(0, len(keys), size=len(keys)), minlength=len(keys))


def test_loaded_posterior_retains_bootstrap_frame_state(tmp_path) -> None:
    agents, decks = registries()
    rows = _f11_rows()
    frame = {("cluster", f"b{index}") for index in range(len(rows))}
    strata = {
        "custom": {
            key: ("even" if index % 2 == 0 else "odd") for index, key in enumerate(sorted(frame))
        }
    }
    post = fit(
        GameTable.from_records(rows),
        agents,
        decks,
        cluster_strata=strata,
    )
    path = tmp_path / "posterior.podrank"
    post.save(path)
    loaded = Posterior.load(path)

    assert loaded.cluster_strata == strata
    assert loaded.cluster_table == post.cluster_table
    assert loaded.outcome_rates() == post.outcome_rates()
    assert np.array_equal(
        _seeded_cluster_weights(loaded, 12345),
        _seeded_cluster_weights(post, 12345),
    )
    assert loaded.patterns.compact is not None
    assert post.patterns.compact is not None
    for name in (
        "cluster_counts",
        "cluster_indices",
        "cluster_patterns",
        "cluster_winner_seats",
    ):
        assert np.array_equal(
            getattr(loaded.patterns.compact, name),
            getattr(post.patterns.compact, name),
        )


def test_posterior_load_rejects_inconsistent_compact_state(tmp_path) -> None:
    post = _posterior()
    path = tmp_path / "posterior.podrank"
    post.save(path)
    payload, arrays = read_posterior_archive(path)
    arrays["compact_agent_indices"][0][0] = 9999
    write_posterior_archive(path, payload, arrays)
    with pytest.raises(PersistenceError, match="agent_indices"):
        Posterior.load(path)


def _rewrite_manifest(path, mutator) -> None:
    with zipfile.ZipFile(path, mode="r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    mutator(manifest["payload"]["arrays"])
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda arrays: arrays["mode"].update(dtype="<i4"), "dtype mismatch"),
        (lambda arrays: arrays["mode"].update(shape=[1]), "shape mismatch"),
        (
            lambda arrays: arrays["mode"].update(shape=[10**9, 10**9]),
            "exceeds the size bound",
        ),
        (
            lambda arrays: arrays["mode"].update(member="arrays/nope.npy"),
            "invalid member name",
        ),
    ],
)
def test_posterior_load_validates_array_descriptors(tmp_path, mutation, message: str) -> None:
    post = _posterior()
    path = tmp_path / "posterior.podrank"
    post.save(path)
    _rewrite_manifest(path, mutation)
    with pytest.raises(PersistenceError, match=message):
        Posterior.load(path)


def test_stratum_crossings_survive_archive_and_resolve_to_keys(tmp_path) -> None:
    agents, decks = registries()
    rows = [
        make_game("g1", suite="train", winner_seat=0, cluster_id="bx"),
        make_game(
            "g2", played_at="2026-01-01T00:00:01Z", suite="eval", winner_seat=1, cluster_id="bx"
        ),
    ]
    strata = {"custom": {("cluster", "bx"): "only"}}
    post = fit(
        GameTable.from_records(rows),
        agents,
        decks,
        cluster_strata=strata,
    )
    assert post.crossing_keys("suite") == (("cluster", "bx"),)
    assert post.crossing_keys("engine_version") == ()
    assert post.crossing_keys("custom") == ()

    path = tmp_path / "posterior.podrank"
    post.save(path)
    loaded = Posterior.load(path)
    assert loaded.stratum_crossings == post.stratum_crossings
    assert loaded.crossing_keys("suite") == (("cluster", "bx"),)
    assert loaded.crossing_keys("engine_version") == ()
    assert loaded.crossing_keys("custom") == ()
    assert loaded.outcome_rates() == post.outcome_rates()
    assert np.array_equal(
        _seeded_cluster_weights(loaded, 99),
        _seeded_cluster_weights(post, 99),
    )


def test_archive_loader_limits(tmp_path, monkeypatch) -> None:
    post = _posterior()
    path = tmp_path / "posterior.podrank"
    post.save(path)
    monkeypatch.setattr(_archive, "MAX_MEMBERS", 1)
    with pytest.raises(PersistenceError, match="member-count"):
        Posterior.load(path)
    monkeypatch.setattr(_archive, "MAX_MEMBERS", 256)
    monkeypatch.setattr(_archive, "MAX_TOTAL_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(PersistenceError, match="total uncompressed"):
        Posterior.load(path)


def test_dense_reference_posterior_save_is_rejected_before_writing(tmp_path) -> None:
    post = _posterior()
    compact = post.patterns.compact
    assert compact is not None
    patterns = []
    for position in range(compact.n_patterns):
        agent_ids = tuple(
            post.parameter_index.supported_agents[int(value)]
            for value in compact.agent_indices[position]
        )
        deck_ids = tuple(
            post.parameter_index.supported_decks[int(value)]
            for value in compact.deck_indices[position]
        )
        swaps = (
            (0, 0, 0, 0)
            if compact.swap_counts is None
            else tuple(int(value) for value in compact.swap_counts[position])
        )
        patterns.append(
            Pattern(
                key=(agent_ids, deck_ids, swaps),
                design=design_row(agent_ids, deck_ids, swaps, post.parameter_index, post._decks),
                target=compact.targets[position],
                weight=int(compact.targets[position].sum()),
            )
        )
    dense_post = dataclasses.replace(
        post,
        patterns=PatternTable(
            index=post.parameter_index,
            patterns=tuple(patterns),
            cluster_counts=MappingProxyType({}),
            n_games=post.patterns.n_games,
        ),
    )
    path = tmp_path / "dense.podrank"
    with pytest.raises(PersistenceError, match="compact"):
        dense_post.save(path)
    assert not path.exists()


def test_posterior_archive_writer_rejects_bad_inputs(tmp_path) -> None:
    path = tmp_path / "posterior.podrank"
    with pytest.raises(PersistenceError, match="invalid array name"):
        write_posterior_archive(path, {}, {"bad/name": np.zeros(1)})
    with pytest.raises(PersistenceError, match="unsupported dtype"):
        write_posterior_archive(path, {}, {"bad": np.array([object()], dtype=object)})
    with pytest.raises(PersistenceError, match="could not write"):
        write_posterior_archive(tmp_path / "missing" / "posterior.podrank", {}, {})


def test_posterior_archive_reader_rejects_bad_manifest_json(tmp_path) -> None:
    path = tmp_path / "posterior.podrank"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            "manifest.json",
            '{"kind":"podrank.posterior","kind":"other",'
            '"schema_version":2,"package_version":"x","payload":{}}',
        )
    with pytest.raises(PersistenceError, match="duplicate JSON key"):
        Posterior.load(path)

    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("arrays/mode.npy", b"")
    with pytest.raises(PersistenceError, match="no manifest"):
        Posterior.load(path)


def test_posterior_load_rejects_extra_archive_member(tmp_path) -> None:
    post = _posterior()
    path = tmp_path / "posterior.podrank"
    post.save(path)
    with zipfile.ZipFile(path, mode="a") as archive:
        archive.writestr("junk.txt", b"nope")
    with pytest.raises(PersistenceError, match="members do not match"):
        Posterior.load(path)


def test_posterior_load_rejects_bad_archive_manifests(tmp_path) -> None:
    path = tmp_path / "posterior.podrank"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PersistenceError):
        Posterior.load(path)

    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "kind": "podrank.posterior",
                    "schema_version": 1,
                    "package_version": "x",
                    "payload": {},
                }
            ),
        )
    with pytest.raises(PersistenceError, match="schema version"):
        Posterior.load(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, mode="w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("manifest.json", "{}")
    with pytest.raises(PersistenceError, match="duplicate members"):
        Posterior.load(path)
