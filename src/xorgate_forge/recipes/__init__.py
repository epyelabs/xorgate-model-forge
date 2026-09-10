"""Recipe registry (`xmf recipes`). Registered like the engine's node packs:
a dict of id to class; the trainer's heavy imports stay inside the recipe
module's methods so `xmf recipes` and `xmf doctor` run on the base install."""

from __future__ import annotations

from .base import ExportOutcome, Preset, Recipe, TrainContext, TrainOutcome
from .yolov8 import Yolov8Recipe

RECIPES: dict[str, type[Recipe]] = {Yolov8Recipe.id: Yolov8Recipe}


def get_recipe(recipe_id: str) -> Recipe:
    cls = RECIPES.get(recipe_id)
    if cls is None:
        raise KeyError(f'recipe "{recipe_id}" is not one of {sorted(RECIPES)}')
    return cls()


def list_recipes() -> list[Recipe]:
    return [cls() for cls in RECIPES.values()]


__all__ = [
    "RECIPES",
    "ExportOutcome",
    "Preset",
    "Recipe",
    "TrainContext",
    "TrainOutcome",
    "Yolov8Recipe",
    "get_recipe",
    "list_recipes",
]
