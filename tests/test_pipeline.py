import os

from common import head_label, read_labels, write_labels
from scrape import parse_page

HERE = os.path.dirname(__file__)


def test_parse_page():
    with open(os.path.join(HERE, 'shot_cafe_page.html'), encoding='utf-8') as f:
        stills = parse_page(f.read())
    assert len(stills) == 16
    assert stills[0] == {
        'id': '18856',
        'img_url': 'https://shot.cafe/images/o/pig-2021-273-18856.jpg',
        'film': 'Pig (2021)',
    }


def test_head_label():
    assert head_label({'close-up', 'low-angle'}, 'shot_type') == 'close-up'
    assert head_label({'close-up', 'low-angle'}, 'angle') == 'low-angle'
    assert head_label({'wide'}, 'shot_type') == 'wide shot'  # alias
    assert head_label({'close-up', 'wide shot'}, 'shot_type') is None  # ambiguous
    assert head_label({'night'}, 'angle') is None


def test_labels_round_trip(tmp_path):
    path = str(tmp_path / 'labels.csv')
    rows = {'7': {'id': '7', 'file': '7.jpg', 'film': 'Heat (1995)', 'tags': {'wide shot', 'overhead'}}}
    write_labels(rows, path)
    assert read_labels(path) == rows
