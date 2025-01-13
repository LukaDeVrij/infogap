import spacy
from transliterate import translit

def get_pron(sents, lang="en"):
    lang_prons = {"en": ("he", "she"), "es": (
        "él", "ella"), "ru": ("он", "она")}
    lang_possessive_prons = {"en": ("his", "her"), "es": (
        "su", "su"), "ru": ("егó", "её")}
    pron_male, pron_female = lang_prons[lang]
    pron_possessive_male, pron_possessive_female = lang_possessive_prons[lang]
    all_words = []
    for sent in sents:
        for w in sent.split():
            all_words.append(w.lower())
    num_he, num_she = all_words.count(pron_male), all_words.count(pron_female)
    if num_he > num_she:
        return (pron_male, pron_possessive_male)
    else:
        return (pron_female, pron_possessive_female)

def get_names(key, sents, lang):
    pron, pron_possessive = get_pron(sents, lang)
    names = [n.strip() for n in key.lower().replace("(", "_").split("_")]+[n.strip()
                                                                            for n in key.lower().replace("(", "_").split("_")]+[pron, pron_possessive]
    if lang == "ru":
        names += [translit(n.strip(), 'ru') for n in key.lower().replace("(", "_").split("_")]
    return names

def get_parsed_mentions(nlp, sents, key, lang):
    sents_parsed = []
    for sent in nlp.pipe(iter(sents), n_threads=24, batch_size=96):
        sent_parsed = []
        for w in sent:
            sent_parsed.append((w.text, w.dep_, w.tag_))
        sents_parsed.append(sent_parsed)
    mentions = get_mentions(key, sents, sents_parsed, lang)
    return mentions

def get_mentions(key, sents, sents_parsed, lang):
    names = get_names(key, sents, lang)
    def _name_in_sent(cur_sent_parsed):
        # TODO: test this, it might not work.
        for w in cur_sent_parsed:
            if w[0] in names:
                return True
        return False
    
    mention_sentences = []
    for i in range(len(sents_parsed)):
        if not _name_in_sent(sents_parsed[i]):
            continue
        else:
            mention_sentences.append(sents[i])
    return mention_sentences

def load_spacy_parsers():
    nlp = {}
    nlp["en"] = spacy.load('en_core_web_sm', pipeline=["tagger", "parser"])
    nlp["ru"] = spacy.load('ru2', pipeline=["tagger", "parser"])
    nlp["es"] = spacy.load("es_core_news_sm", pipeline=["tagger", "parser"])
    # add korean 
    nlp["ko"] = spacy.load("ko_core_news_sm", pipeline=["tagger", "parser"])
    nlp["fr"] = spacy.load("fr_core_news_sm", pipeline=["tagger", "parser"])
    return nlp

class SpacyParser:
    parsers = {}

    @classmethod
    def get_parser(cls, lang):
        if cls.parsers == {}:
            cls.parsers = load_spacy_parsers()
        return cls.parsers[lang]