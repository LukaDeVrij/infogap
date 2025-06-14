from datetime import datetime
import pytest
from scrap_living_people import get_edit_history, parse_wikipedia_korean_datetime_format, convert_edit_diff_to_int
from utils import get_edit_history_metadata_all_languages, get_edit_history_metadata
from constants import TARGET_LANGUAGES

def test_get_edit_history():
    person_id = "Scottie_Barnes"
    weekly_snapshots = get_edit_history(person_id)

def test_get_edit_history_all_languages():
    person_id = "Scottie_Barnes"
    lang_to_metadata = get_edit_history_metadata_all_languages(person_id)

    assert len(lang_to_metadata['enwiki']) > len(lang_to_metadata['frwiki'])
    # print the lengths of the edit histories for each language
    for lang, metadata in lang_to_metadata.items():
        print(f"{lang}: {len(metadata)}")

def test_get_edit_history_w_korean():
    person_id = "Jennie_(singer)"
    weekly_snapshots = get_edit_history(person_id, TARGET_LANGUAGES)

def test_get_edit_history_3():
    person_id = "John_Alcorn_(singer)"
    weekly_snapshots = get_edit_history(person_id, TARGET_LANGUAGES)

def test_get_edit_history_metadata():
    history_metadata = get_edit_history_metadata("en.wikipedia.org", "Scottie_Barnes")
    assert history_metadata[0].diff_bytes == '+43'

def test_parse_wikipedia_korean_datetime_format():
    assert parse_wikipedia_korean_datetime_format('2021년 9월 5일 (일) 04:09') == datetime(2021, 9, 5, 4, 9)

def test_convert_edit_diff_to_int():
    assert convert_edit_diff_to_int('+43') == 43
    assert convert_edit_diff_to_int('-43') == 43
    assert convert_edit_diff_to_int('0') == 0