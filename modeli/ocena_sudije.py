from pydantic import BaseModel, ConfigDict, Field

class OcenaSudije(BaseModel):
    """Strukturisana ocena koju LLM sudija daje za jedan odgovor chatbota."""

    model_config = ConfigDict(frozen=True) #da se ne dozvoli promena ocene koju je dao sudija

    analiza: str = Field(
        description="Kratka analiza odgovora (2-3 recenice). "
                    "Obavezno navedi nedostatke ako postoje. "       
    )
    relevantnost: int = Field(
        ge=1, le=5,
        description="Da li odgovor direktno adresira pitanje? Ako je pitanje van "
                    "teme, da li ga je chatbot ispravno prepoznao kao takvo?"
    )
    utemeljenost: int = Field(
        ge=1, le=5,
        description="Da li je svaka tvrdnja potkrepljena kontekstom? Ako postoji "
                    "i najmanja informacija koje nema u kontekstu, najviše 2."
    )
    kompletnost: int = Field(
        ge=1, le=5,
        description="Da li odgovor pokriva sve bitne aspekte pitanja dostupne u kontekstu?"
    )
    jezik: int = Field(
        ge=1, le=5,
        description="Da li je odgovor gramatički ispravan, jasan, prirodan srpski?"
    )
    obrazlozenje: str = Field(
        description="Jedna kratka rečenica koja sumira zašto su date ove ocene."
    )