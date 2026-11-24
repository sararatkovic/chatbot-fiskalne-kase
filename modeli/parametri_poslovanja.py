from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ParametriPoslovanja(BaseModel):
    """Parametri poslovanja koje je korisnik naveo, na osnovu kojih se preporučuje kasa."""

    model_config = ConfigDict(frozen=True)

    delatnost: Optional[str] = Field(
        default=None,
        description="Čime se korisnik bavi, npr. 'restoran', 'pekara', 'dostava hrane'.",
    )
    promet: Optional[Literal["mali", "veliki"]] = Field(
        default=None,
        description="Obim prometa: 'veliki' ako korisnik ima gužvu, mnogo računa dnevno "
                    "ili veliki broj artikala; 'mali' ako je promet umeren ili manji.",
    )
    mobilnost: Optional[Literal["da", "ne", "nije bitno"]] = Field(
        default=None,
        description="Da li je korisniku potrebna mobilna kasa za naplatu van pulta.",
    )
    placanje_karticom: Optional[Literal["da", "ne", "nije bitno"]] = Field(
        default=None,
        description="Da li korisnik želi da kupci plaćaju karticom direktno na kasi.",
    )