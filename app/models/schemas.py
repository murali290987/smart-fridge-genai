"""Pydantic models shared across the vision, inventory, and meal-time modules."""
from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.config import VISION_CONFIDENCE_THRESHOLD


class Ingredient(BaseModel):
    name: str
    estimated_quantity: str
    unit: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_confirmation: bool = False

    @model_validator(mode="after")
    def flag_low_confidence(self) -> "Ingredient":
        # Never silently drop low-confidence ingredients -- flag them instead.
        if self.confidence < VISION_CONFIDENCE_THRESHOLD:
            self.needs_confirmation = True
        return self


class IngredientDetectionResponse(BaseModel):
    ingredients: list[Ingredient]


class InventoryRecord(BaseModel):
    """A single row from the `inventory` table (PostgreSQL)."""

    id: int
    name: str
    estimated_quantity: str
    unit: str
    confidence: float
    needs_confirmation: bool
    source: str
    created_at: dt.datetime
    updated_at: dt.datetime


class MealPeriod(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    snacks = "snacks"
    dinner = "dinner"
    other = "other"


class MealPeriodInfo(BaseModel):
    meal_period: MealPeriod
    current_time: str
    timezone: str = "local"
