
import pandas as pd
import re
import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger_eng')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("1_reddit_dataset_uncleaned.csv")
df['clean_text'] = df['text'].apply(clean_text)

hyperbolic_words = ["always", "never", "everyone", "no one", "completely", "absolutely", "undeniable", "worst", "best",
    "total", "utter", "entire", "disaster", "catastrophe", "massive", "epic", "unthinkable", "incredible",
    "ridiculous", "explosive", "insane", "shocking", "monumental", "giant", "colossal"]
emotional_words = ["angry", "furious", "love", "hate", "disgusted", "thrilled", "terrified", "happy", "sad",
    "outraged", "grateful", "heartbroken", "ashamed", "anxious", "panicked", "hopeful", "devastated",
    "excited", "scared", "enraged", "lonely", "helpless", "nervous", "shame"]
vague_sources = ["experts say", "sources claim", "reportedly", "many believe", "some argue", "it seems", "analysts suggest",
    "studies show", "people say", "as per reports", "unnamed sources", "officials say", "there are reports",
    "according to some", "insiders reveal", "word on the street", "rumors suggest", "they say", "it is claimed",
    "some experts believe"]
clickbait_phrases = ["you won't believe", "shocking", "what happened next", "goes viral", "will blow your mind",
    "this is what happens", "can't handle", "top secret", "exposed", "unbelievable", "watch now",
    "number x will surprise you", "everything you need to know", "this one simple trick", "before it's too late",
    "the truth about", "revealed", "click here", "must see", "don't miss", "it gets worse"]
hedging_words = ["might", "could", "possibly", "allegedly", "reportedly", "suggests", "it seems", "some believe",
    "was said to", "was believed to", "it is likely", "there is a chance", "it is suggested", "apparently",
    "it is assumed", "is being considered", "there are concerns", "experts are unsure"]
bias_words = ["clearly", "undeniably", "of course", "naturally", "obviously", "it's a fact", "it’s clear that",
    "without a doubt", "certainly", "unquestionably", "truly", "in fact", "as everyone knows", "everyone agrees",
    "very", "so", "really", "extremely", "highly", "deeply", "strongly", "inherently", "completely"]
negation_words = ["not", "never", "no", "none", "without"]
superlatives = ["best", "worst", "most", "least", "greatest"]
certainty_words = ["must", "should", "certainly", "undoubtedly", "surely"]
quantifier_words = ["everyone", "nobody", "thousands", "millions", "countless"]

def contains_marker(text, marker_list):
    return int(any(marker in text for marker in marker_list))

df['has_hyperbole'] = df['clean_text'].apply(lambda x: contains_marker(x, hyperbolic_words))
df['has_emotion'] = df['clean_text'].apply(lambda x: contains_marker(x, emotional_words))
df['has_vague_source'] = df['clean_text'].apply(lambda x: contains_marker(x, vague_sources))
df['has_clickbait'] = df['clean_text'].apply(lambda x: contains_marker(x, clickbait_phrases))
df['has_hedging'] = df['clean_text'].apply(lambda x: contains_marker(x, hedging_words))
df['has_bias_word'] = df['clean_text'].apply(lambda x: contains_marker(x, bias_words))
df['has_negation'] = df['clean_text'].apply(lambda x: contains_marker(x, negation_words))
df['has_superlative'] = df['clean_text'].apply(lambda x: contains_marker(x, superlatives))
df['has_certainty'] = df['clean_text'].apply(lambda x: contains_marker(x, certainty_words))
df['has_quantifier'] = df['clean_text'].apply(lambda x: contains_marker(x, quantifier_words))

df['char_length'] = df['clean_text'].apply(len)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
df['avg_word_length'] = df['clean_text'].apply(lambda x: (sum(len(word) for word in x.split()) / len(x.split())) if len(x.split()) > 0 else 0)

df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']

# POS features
def extract_pos_features(text):
    tokens = word_tokenize(text, preserve_line=True)
    tags = pos_tag(tokens)

    modal_count = sum(1 for word, tag in tags if tag == 'MD')
    adj_noun_pairs = sum(1 for i in range(len(tags) - 1) if tags[i][1] == 'JJ' and tags[i + 1][1].startswith('NN'))
    passive_count = sum(1 for i in range(len(tags) - 1)
                        if tags[i][0].lower() in ['is', 'was', 'were', 'are', 'been', 'be', 'being']
                        and tags[i + 1][1] == 'VBN')

    return pd.Series({
        'modal_count': modal_count,
        'adj_noun_pairs': adj_noun_pairs,
        'passive_voice_count': passive_count
    })

df[['modal_count', 'adj_noun_pairs', 'passive_voice_count']] = df['clean_text'].apply(extract_pos_features)

df.to_csv("2_reddit_dataset_preprocessed.csv", index=False)
print("Dataset saved as 2_reddit_dataset_preprocessed.csv")
