from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Klasifikacija(BaseModel):
    """Klasifikacija pitanja koje korisnik postavlja."""

    model_config = ConfigDict(frozen=True)

    tip: Literal["preporuka", "kvar", "van_teme"] = Field(
        description="Korisnik postavlja pitanje za preporuku kase, resavanje kvara na kasi ili van teme."
    )