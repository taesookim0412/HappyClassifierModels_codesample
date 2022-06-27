import re

def _basic_english_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for En glish words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    _patterns = [r'\'',
                 r'\"',
                 r'\.',
                 r'<br \/>',
                 r',',
                 r'\(',
                 r'\)',
                 r'\!',
                 r'\?',
                 r'\;',
                 r'\:',
                 r'\s+']

    _replacements = [' \'  ',
                     '',
                     ' . ',
                     ' ',
                     ' , ',
                     ' ( ',
                     ' ) ',
                     ' ! ',
                     ' ? ',
                     ' ',
                     ' ',
                     ' ']

    _patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()

# Usage:
# tokenizer = lambda line: _basic_english_normalize(line)



def create_vocab(word_frequencies: dict[str, int], specials, min_freq = 1, special_first=True):
    '''
    Removes specials and puts them at the front or the beginning.
    Filters out words that do not fill min_freq requirements.
    :param word_frequencies: map of {word: freq}
    :param specials: list of ['<unk>'] specials
    :param min_freq: minimum frequency the word has to appear in our vocabulary
    :param special_first: whether specials are most common in our vocab or not.
    :return: dict of { word: freq }
    '''
    tokens = []

    if special_first:
        tokens.extend(specials)

    specials_set = set(specials)

    for word, freq in word_frequencies.items():
        if freq >= min_freq and word not in specials_set:
            tokens.append(word)

    if special_first is False:
        tokens.extend(specials)

    res = {}
    for i, token in enumerate(tokens):
        res[token] = i

    return res

def build_vocab_from_iterator_custom(normalized_sentences_list: list[list[str]], specials: list[str] = ["<unk>"]):
    '''
    Returns a map of {token: freq}. Depending on if we specify special first and min frequency we obtain a different result map.
    :param normalized_sentences_list: List of sentences that have been tokenized. For example, [['this', ',' 'sentence'], ['hello',',','world']]
    :param specials: list of specials
    :return: map of { token : freq }
    '''
    word_frequencies = {}
    for sentence in normalized_sentences_list:
        for word in sentence:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # sort by descending frequencies then lexicographically.
    word_frequencies = dict(sorted(word_frequencies.items(), key=lambda x: (-1 * x[1], x[0])))

    return create_vocab(word_frequencies, specials)

# Usage:
# build_vocab_from_iterator_custom(list(map(lambda k: tokenizer(k), [txt for txt, label in train_ds.train_data])), specials=["<unk>"])

def build_vocab_from_tokenized_sentences_optimized(normalized_sentences_list: list[list[str]], specials: list[str] = ["<unk>"]):
    word_frequencies = {}
    specials_set = set(specials)
    max_freq = -1
    for sentence in normalized_sentences_list:
        for word in sentence:
            if word not in specials_set:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
                max_freq = max(max_freq, word_frequencies[word])
    biggest_freq_after = max_freq + 1
    for special in reversed(specials):
        word_frequencies[special] = biggest_freq_after
        biggest_freq_after += 1

    word_frequencies_sorted = {k: v for k, v in sorted(word_frequencies.items(), key=lambda x: (-x[1], x[0]))}
    res = {}
    for i, (word, freq) in enumerate(word_frequencies_sorted.items()):
        res[word] = i
    return res

# Usage:
# build_vocab_from_tokenized_sentences_optimized(list(map(lambda k: tokenizer(k), [txt for txt, label in train_ds.train_data])), specials=["<unk>"])

def vocab_from_custom_lambda(vocab_map, tokenized):
    res = []
    for token in tokenized:
        res.append(vocab_map.get(token, 0))
    return res

# Usage:
# vocab_from_custom_optimized = build_vocab_from_tokenized_sentences_optimized(list(map(lambda k: _basic_english_normalize(k), [txt for txt, label in train_ds.train_data])))
# vocab_from_custom_optimized_pipeline = lambda sentence: vocab_from_custom_lambda(vocab_from_custom_optimized, tokenizer(sentence))

#Create a test method in similar syntax to java
def test_pipelines(pt_pipeline, deciphered_pipeline, ds):
    accuracy = 0
    total_count = 0
    for i, (text, label) in enumerate(ds.train_data):
        pt_text = pt_pipeline(text)
        new_text = deciphered_pipeline(text)

        # assert pt_text array equals new_text array
        accuracy += (pt_text == new_text)
        #print(pt_text, new_text)
        total_count += 1
        if i == 0:
            print(new_text)

        if (i + 1) % (len(ds.train_data) // 5) == 0:
            print(f"Iteration {i} | Accuracy: {(accuracy / total_count) * 100} %.")
        if accuracy != total_count:
            print(pt_text, new_text)
            break
    try:
        assert(accuracy == total_count)
    except AssertionError:
        print("Not the same")

#Usage : test_vocab(vocab_pipeline, vocab_from_custom_optimized_pipeline, train_ds)