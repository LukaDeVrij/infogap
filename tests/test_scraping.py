import ipdb
import pytest

from wikipedia_edit_scrape_tool import get_text, Header, Paragraph 
from packages.steps.info_diff_steps import step_retrieve_en_content_blocks

def test_retrieve_en_content_blocks():
    en_bio_id = "Kim_Petras"
    en_content_blocks = step_retrieve_en_content_blocks(en_bio_id)
    assert len(en_content_blocks) > 0
    # filter for paragraphs where the clean_text attribute string has fewer than 6 words.
    short_paragraphs = list(filter(lambda x: isinstance(x, Paragraph) and len(x.clean_text.split()) < 6, en_content_blocks))
    assert len(short_paragraphs) == 0
    ipdb.set_trace()