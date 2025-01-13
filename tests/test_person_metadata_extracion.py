from packages.steps.metadata_extract_steps import get_popularity, get_ethnicity, get_sex_or_gender

def test_person_popularity_index_extraction():
    wikidata_id = "Q167635"
    popularity = get_popularity(wikidata_id)

def test_get_person_ethnicity():
    wikidata_id = "Q167635"
    ethnicity = get_ethnicity(wikidata_id)
    assert ethnicity == "African Americans"

def test_get_person_gender():
    wikidata_id = "Q167635"
    gender = get_sex_or_gender(wikidata_id)
    assert gender == "male"