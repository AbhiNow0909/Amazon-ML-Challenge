import pandas as pd

from business_entity_resolution.normalize import (
    generate_address_views,
    generate_name_views,
    normalize_country,
    normalize_records,
)


def test_name_normalization_accents_connectors_and_legal_forms():
    french = generate_name_views("  Société Générale S.A.S. ")
    assert french["name_full"] == "societe generale s a s"
    # Punctuated S.A.S. remains conservative rather than being guessed as SAS.
    assert french["name_core"] == "societe generale s a s"

    india = generate_name_views("ABC & Sons Private Limited")
    assert india["name_full"] == "abc and sons private limited"
    assert india["name_core"] == "abc and sons"
    assert india["name_sorted"] == "abc and sons"


def test_spacing_punctuation_and_sorted_tokens():
    left = generate_name_views("Traders,   Sharma Pvt. Ltd.")
    right = generate_name_views("Sharma Traders")
    assert left["name_core"] == "traders sharma"
    assert left["name_sorted"] == right["name_sorted"]


def test_address_numbers_and_postcode_are_preserved():
    views = generate_address_views("Plot 12/3, M.G. Road, Bengaluru 560 001")
    assert set(views["address_numeric_tokens"]) >= {"12", "3", "560", "001"}
    assert "12 3" in views["address_full"]


def test_country_is_open_set_and_normalized():
    assert normalize_country("  Nouvelle-Zélande ") == "nouvelle-zélande"
    assert normalize_country("France") == "france"


def test_normalize_records_keeps_raw_columns():
    frame = pd.DataFrame(
        [["S1-1", "Acme, Inc.", "10 Main St.", "US"]],
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    normalized = normalize_records(frame)
    assert normalized.loc[0, "business_name"] == "Acme, Inc."
    assert normalized.loc[0, "name_core"] == "acme"
    assert normalized.loc[0, "address_numeric_tokens"] == "10"

