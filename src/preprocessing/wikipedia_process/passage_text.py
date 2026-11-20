import re

import mwparserfromhell


MIN_PASSAGE_LENGTH = 29
MAX_PASSAGE_LENGTH = 1094


def format_passage_text(data):
    """Build the searchable text, preserving whitespace in the cleaned body."""
    title = " ".join(data["section_titles"])
    title = re.sub(r'_dup\d+', '', title)
    text = mwparserfromhell.parse(data["text"]).strip_code()
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = text.replace("\x00", "")
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return title + "\n" + text


def is_searchable_length(length):
    return MIN_PASSAGE_LENGTH <= length <= MAX_PASSAGE_LENGTH
