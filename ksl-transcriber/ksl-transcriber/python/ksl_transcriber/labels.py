"""Built-in label sets (alphabet, digits, words) plus user-added custom words.

Mirrors js/labels.js in the browser version, feature for feature, so the two
implementations stay in sync. Custom words are stored separately from the
recorded training samples (dataset.py) so resetting recordings never wipes
out the vocabulary someone has built up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CUSTOM_WORDS_FILE = DATA_DIR / "custom_words.json"

ALPHABET: List[str] = [
    'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Қ',
    'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С', 'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х',
    'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'І', 'Ь', 'Э', 'Ю', 'Я'
]

DIGITS: List[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

BUILTIN_WORDS: List[tuple] = [
    ('Сәлем', 'hello'),
    ('Сау болыңыз', 'goodbye'),
    ('Рахмет', 'thank you'),
    ('Иә', 'yes'),
    ('Жоқ', 'no'),
    ('Кешіріңіз', 'sorry / excuse me'),
    ('Көмек', 'help'),
    ('Өтінемін', 'please'),
    ('Мен', 'I / me'),
    ('Сен', 'you'),
    ('Есім', 'name'),
    ('Жақсы', 'good / fine'),
    ('Жаман', 'bad'),
    ('Су', 'water'),
    ('Тамақ', 'food'),
    ('Үй', 'home'),
    ('Дос', 'friend'),
    ('Отбасы', 'family'),
    ('Уақыт', 'time'),
    ('Қалайсың', 'how are you'),
]

CAT_ORDER = ['alphabet', 'numbers', 'words']


@dataclass(frozen=True)
class LabelItem:
    label: str
    gloss: str
    buffered: bool   # True = accumulates into the spelling buffer (letters/digits)
    custom: bool = False


class LabelStore:
    """Holds the current category -> [LabelItem] mapping, including custom words."""

    def __init__(self):
        self._custom_words: List[dict] = self._load_custom_words()
        self.categories = self._build_categories()

    # -- persistence --------------------------------------------------
    def _load_custom_words(self) -> List[dict]:
        try:
            if CUSTOM_WORDS_FILE.exists():
                data = json.loads(CUSTOM_WORDS_FILE.read_text(encoding='utf-8'))
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _save_custom_words(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CUSTOM_WORDS_FILE.write_text(
            json.dumps(self._custom_words, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    # -- building the live category map --------------------------------
    def _build_categories(self):
        return {
            'alphabet': {
                'name': 'Alphabet',
                'items': [LabelItem(ch, ch, buffered=True) for ch in ALPHABET],
            },
            'numbers': {
                'name': 'Numbers',
                'items': [LabelItem(d, d, buffered=True) for d in DIGITS],
            },
            'words': {
                'name': 'Words',
                'items': (
                    [LabelItem(kk, gloss, buffered=False) for kk, gloss in BUILTIN_WORDS]
                    + [LabelItem(w['kk'], w.get('gloss', w['kk']), buffered=False, custom=True)
                       for w in self._custom_words]
                ),
            },
        }

    # -- public API ------------------------------------------------------
    def add_word(self, kk: str, gloss: str = '') -> tuple:
        """Returns (ok: bool, error: str | None)."""
        kk = (kk or '').strip()
        gloss = (gloss or '').strip()
        if not kk:
            return False, 'Type a word first.'
        if any(item.label == kk for item in self.categories['words']['items']):
            return False, 'That word is already in the list.'
        self._custom_words.append({'kk': kk, 'gloss': gloss or kk})
        self._save_custom_words()
        self.categories = self._build_categories()
        return True, None

    def remove_custom_word(self, kk: str) -> None:
        self._custom_words = [w for w in self._custom_words if w['kk'] != kk]
        self._save_custom_words()
        self.categories = self._build_categories()

    def items(self, category: str) -> List[LabelItem]:
        return self.categories[category]['items']

    def category_name(self, category: str) -> str:
        return self.categories[category]['name']

    def find(self, category: str, label: str) -> LabelItem | None:
        return next((i for i in self.items(category) if i.label == label), None)
