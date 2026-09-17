"""Pydantic models for validating Qwen3-VL's ingredient detection output."""
from __future__ import annotations

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
