"""
Deterministic meal-period detection from local system time.

Intentionally NOT an LLM call -- the current time is a fact, not something
to ask a language model to infer.
"""
from __future__ import annotations

import datetime as dt

from app.models.schemas import MealPeriod, MealPeriodInfo

# (inclusive start, exclusive end), in local 24h time.
MEAL_PERIOD_RANGES: dict[MealPeriod, tuple[dt.time, dt.time]] = {
    MealPeriod.breakfast: (dt.time(6, 0), dt.time(10, 30)),
    MealPeriod.lunch: (dt.time(11, 30), dt.time(15, 0)),
    MealPeriod.snacks: (dt.time(15, 0), dt.time(18, 0)),
    MealPeriod.dinner: (dt.time(18, 0), dt.time(22, 30)),
}


def detect_meal_period(now: dt.datetime | None = None) -> MealPeriodInfo:
    now = now or dt.datetime.now()
    current_time = now.time()

    period = MealPeriod.other
    for candidate, (start, end) in MEAL_PERIOD_RANGES.items():
        if start <= current_time < end:
            period = candidate
            break

    return MealPeriodInfo(meal_period=period, current_time=now.strftime("%H:%M"))
