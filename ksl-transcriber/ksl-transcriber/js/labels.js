/* labels.js
 * Defines the built-in label sets (alphabet, digits, words) and manages
 * user-added custom words. Custom words are stored separately from the
 * recorded training samples so a "reset training data" action never wipes
 * out the vocabulary the user has built up.
 */
window.KSL = window.KSL || {};
(function (KSL) {
  "use strict";

  const ALPHABET = [
    'А','Ә','Б','В','Г','Ғ','Д','Е','Ё','Ж','З','И','Й','К','Қ','Л','М','Н',
    'Ң','О','Ө','П','Р','С','Т','У','Ұ','Ү','Ф','Х','Һ','Ц','Ч','Ш','Щ','Ъ',
    'Ы','І','Ь','Э','Ю','Я'
  ];

  const DIGITS = ['0','1','2','3','4','5','6','7','8','9'];

  // [Kazakh word, English gloss]
  const BUILTIN_WORDS = [
    ['Сәлем', 'hello'],
    ['Сау болыңыз', 'goodbye'],
    ['Рахмет', 'thank you'],
    ['Иә', 'yes'],
    ['Жоқ', 'no'],
    ['Кешіріңіз', 'sorry / excuse me'],
    ['Көмек', 'help'],
    ['Өтінемін', 'please'],
    ['Мен', 'I / me'],
    ['Сен', 'you'],
    ['Есім', 'name'],
    ['Жақсы', 'good / fine'],
    ['Жаман', 'bad'],
    ['Су', 'water'],
    ['Тамақ', 'food'],
    ['Үй', 'home'],
    ['Дос', 'friend'],
    ['Отбасы', 'family'],
    ['Уақыт', 'time'],
    ['Қалайсың', 'how are you']
  ];

  const CUSTOM_WORDS_KEY = 'kslCustomWords_v1';

  function loadCustomWords() {
    try {
      const raw = localStorage.getItem(CUSTOM_WORDS_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      console.warn('Could not load custom words', e);
      return [];
    }
  }
  function saveCustomWords(list) {
    try { localStorage.setItem(CUSTOM_WORDS_KEY, JSON.stringify(list)); }
    catch (e) { console.warn('Could not save custom words', e); }
  }

  let customWords = loadCustomWords(); // [{kk, gloss}]

  function buildCategories() {
    return {
      alphabet: {
        name: 'Alphabet',
        items: ALPHABET.map(ch => ({ label: ch, gloss: ch, buffered: true, custom: false }))
      },
      numbers: {
        name: 'Numbers',
        items: DIGITS.map(d => ({ label: d, gloss: d, buffered: true, custom: false }))
      },
      words: {
        name: 'Words',
        items: [
          ...BUILTIN_WORDS.map(([kk, gloss]) => ({ label: kk, gloss, buffered: false, custom: false })),
          ...customWords.map(w => ({ label: w.kk, gloss: w.gloss, buffered: false, custom: true }))
        ]
      }
    };
  }

  KSL.ALPHABET = ALPHABET;
  KSL.DIGITS = DIGITS;
  KSL.CAT_ORDER = ['alphabet', 'numbers', 'words'];
  KSL.CATEGORIES = buildCategories();

  /**
   * Add a new custom word to the Words category.
   * Returns {ok:true} on success or {ok:false, error} if rejected.
   */
  KSL.addWord = function (kk, gloss) {
    kk = (kk || '').trim();
    gloss = (gloss || '').trim();
    if (!kk) return { ok: false, error: 'Type a word first.' };
    const already = KSL.CATEGORIES.words.items.some(i => i.label === kk);
    if (already) return { ok: false, error: 'That word is already in the list.' };
    customWords.push({ kk, gloss: gloss || kk });
    saveCustomWords(customWords);
    KSL.CATEGORIES = buildCategories();
    return { ok: true };
  };

  /**
   * Remove a custom word (built-in words cannot be removed this way).
   */
  KSL.removeCustomWord = function (kk) {
    customWords = customWords.filter(w => w.kk !== kk);
    saveCustomWords(customWords);
    KSL.CATEGORIES = buildCategories();
  };

})(window.KSL);
