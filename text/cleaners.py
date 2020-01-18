# coding: utf-8

# Code based on https://github.com/keithito/tacotron/blob/master/text/cleaners.py

import re
from .korean import tokenize as ko_tokenize

# # Added to support LJ_speech
# from unidecode import unidecode
# from .en_numbers import normalize_numbers as en_normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def korean_cleaners(text):
    '''Pipeline for Korean text, including number and abbreviation expansion.'''
    text = ko_tokenize(text)  # '존경하는' --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']
    return text


# # List of (regular expression, replacement) pairs for abbreviations:
# _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
#     ('mrs', 'misess'),
#     ('mr', 'mister'),
#     ('dr', 'doctor'),
#     ('st', 'saint'),
#     ('co', 'company'),
#     ('jr', 'junior'),
#     ('maj', 'major'),
#     ('gen', 'general'),
#     ('drs', 'doctors'),
#     ('rev', 'reverend'),
#     ('lt', 'lieutenant'),
#     ('hon', 'honorable'),
#     ('sgt', 'sergeant'),
#     ('capt', 'captain'),
#     ('esq', 'esquire'),
#     ('ltd', 'limited'),
#     ('col', 'colonel'),
#     ('ft', 'fort'),
# ]]


# def expand_abbreviations(text):
#     for regex, replacement in _abbreviations:
#         text = re.sub(regex, replacement, text)
#     return text
#
#
# def expand_numbers(text):
#     return en_normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


# def convert_to_ascii(text):
#     return unidecode(text)


def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


# def transliteration_cleaners(text):
#     # text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = collapse_whitespace(text)
#     return text
#
#
# def english_cleaners(text):
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_numbers(text)
#     text = expand_abbreviations(text)
#     text = collapse_whitespace(text)
#     return text
