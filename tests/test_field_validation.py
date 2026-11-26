"""Field context and recipe validation branches."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from podrank import DeckRegistry
from podrank.errors import NoEligiblePodsError
from podrank.rating.field import Field


def _decks(counts: dict[int, int]) -> DeckRegistry:
    rows = [
        {
            "deck_id": f"b{bracket}d{index}",
            "bracket": bracket,
            "split": "train",
            "deck_version": f"b{bracket}d{index}v1",
        }
        for bracket, count in counts.items()
        for index in range(count)
    ]
    return DeckRegistry.from_records(rows)


def _context() -> dict[str, object]:
    return {
        "decks": ("b0d0", "b0d1", "b0d2", "b0d3"),
        "swap_counts": (0, 0, 0, 0),
        "opponents": ("a", "b", "c"),
        "focal_seat": 0,
        "weight": 1.0,
    }


@pytest.mark.parametrize(
    ("context", "match"),
    [
        (3, "FieldContext, mapping"),
        ({**_context(), "extra": 1}, "unknown field context"),
        (
            {key: value for key, value in _context().items() if key != "decks"},
            "missing field context",
        ),
        ({**_context(), "decks": "not-a-sequence"}, "decks must be a sequence"),
        ({**_context(), "opponents": "not-a-sequence"}, "opponents must be a sequence"),
        ({**_context(), "swap_counts": "not-a-sequence"}, "swap_counts must be a sequence"),
        ({**_context(), "decks": ("d0", "d1", "d2")}, "decks must have length 4"),
        ({**_context(), "opponents": ("a", "b")}, "opponents must have length 3"),
        ({**_context(), "swap_counts": (0, 0)}, "swap_counts must have length 4"),
        ({**_context(), "swap_counts": (0, -1, 0, 0)}, "swap_counts must be >= 0"),
        ({**_context(), "focal_seat": 4}, "focal_seat must be 0..3"),
        ({**_context(), "focal_seat": True}, "focal_seat must be an integer"),
        ({**_context(), "weight": 0}, "weight must be a positive finite number"),
        ({**_context(), "weight": "heavy"}, "weight must be a positive finite number"),
    ],
)
def test_field_context_validation(context: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Field.from_contexts("f", 1, [context])


def test_field_requires_a_context() -> None:
    with pytest.raises(ValueError, match="at least one context"):
        Field.from_contexts("f", 1, [])


def _base_recipe() -> dict[str, object]:
    return {
        "field_id": "f",
        "version": 1,
        "recipe": {
            "deck_split": "train",
            "pod": {
                "bracket_policy": "same_bracket",
                "bracket_weights": "equal",
                "min_decks_per_bracket": 4,
                "distinct_decks": True,
                "swap_counts": 0,
            },
            "opponents": {
                "pool": ["a", "b", "c"],
                "draw": "independent_uniform_per_seat",
            },
            "focal_seat": "uniform",
            "n_contexts": 2,
            "seed": 1,
        },
    }


def _inner(recipe: dict[str, object]) -> dict[str, object]:
    value = recipe["recipe"]
    assert isinstance(value, dict)
    return value


def _pod(recipe: dict[str, object]) -> dict[str, object]:
    value = _inner(recipe)["pod"]
    assert isinstance(value, dict)
    return value


def _opponents(recipe: dict[str, object]) -> dict[str, object]:
    value = _inner(recipe)["opponents"]
    assert isinstance(value, dict)
    return value


Mutator = Callable[[dict[str, object]], None]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda recipe: recipe.pop("field_id"), "missing recipe keys"),
        (lambda recipe: recipe.update({"extra": 1}), "unknown recipe keys"),
        (lambda recipe: _inner(recipe).pop("seed"), "missing recipe keys"),
        (lambda recipe: _inner(recipe).update({"deck_split": "bad"}), "deck_split"),
        (lambda recipe: _inner(recipe).update({"pod": "bad"}), "pod must be a mapping"),
        (lambda recipe: _inner(recipe).update({"opponents": "bad"}), "opponents must be a mapping"),
        (lambda recipe: _pod(recipe).update({"extra": 1}), "unknown pod keys"),
        (lambda recipe: _pod(recipe).update({"bracket_weights": "weighted"}), "bracket_weights"),
        (lambda recipe: _pod(recipe).update({"distinct_decks": False}), "distinct_decks"),
        (lambda recipe: _pod(recipe).update({"min_decks_per_bracket": 0}), "min_decks_per_bracket"),
        (lambda recipe: _pod(recipe).update({"swap_counts": -1}), "swap_counts must be >= 0"),
        (lambda recipe: _opponents(recipe).update({"extra": 1}), "unknown opponents keys"),
        (lambda recipe: _opponents(recipe).update({"draw": "bad"}), "opponents.draw"),
        (lambda recipe: _opponents(recipe).update({"pool": "abc"}), "opponents.pool"),
        (
            lambda recipe: _opponents(recipe).update({"pool": []}),
            "opponents.pool must be non-empty",
        ),
        (lambda recipe: _inner(recipe).update({"focal_seat": "seat0"}), "focal_seat"),
        (lambda recipe: _inner(recipe).update({"n_contexts": 0}), "n_contexts"),
        (lambda recipe: _inner(recipe).update({"seed": -1}), "seed"),
    ],
)
def test_materialise_recipe_validation(mutate: Mutator, match: str) -> None:
    recipe = _base_recipe()
    mutate(recipe)
    with pytest.raises(ValueError, match=match):
        Field.materialise(recipe, decks=_decks({0: 4, 1: 3}))


def test_materialise_any_policy_and_bracket_exclusion_warning() -> None:
    recipe = _base_recipe()
    _pod(recipe)["bracket_policy"] = "any"
    field = Field.materialise(recipe, decks=_decks({0: 4, 1: 3}))
    assert field.contexts

    recipe = _base_recipe()
    with pytest.warns(UserWarning, match="excluded"):
        field = Field.materialise(recipe, decks=_decks({0: 4, 1: 3}))
    assert all(context.decks[0].startswith("b0") for context in field.contexts)


def test_materialise_any_policy_requires_four_decks() -> None:
    recipe = _base_recipe()
    _pod(recipe)["bracket_policy"] = "any"
    with pytest.raises(NoEligiblePodsError, match="4 required"):
        Field.materialise(recipe, decks=_decks({0: 3}))
