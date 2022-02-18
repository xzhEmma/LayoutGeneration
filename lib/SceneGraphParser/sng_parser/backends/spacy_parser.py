#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : spacy_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/21/2018
#
# This file is part of SceneGraphParser.
# Distributed under terms of the MIT license.
# https://github.com/vacancy/SceneGraphParser

from .. import database
from ..database import str_insert
from ..parser import Parser
from .backend import ParserBackend
from .English_to_num import spoken_word_to_number
import copy


__all__ = ['SpacyParser']


@Parser.register_backend
class SpacyParser(ParserBackend):
    """
    Scene graph parser based on spaCy.
    """

    __identifier__ = 'spacy'

    def __init__(self, model='en_core_web_sm'):
        """
        Args:
            model (str): a spec for the spaCy model. (default: en). Please refer to the
            official website of spaCy for a complete list of the available models.
            This option is useful if you are dealing with languages other than English.
        """

        self.model = model

        try:
            import spacy
        except ImportError as e:
            raise ImportError('Spacy backend requires the spaCy library. Install spaCy via pip first.') from e

        try:
            self.nlp = spacy.load(model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e

    def parse(self, sentence, return_doc=False):
        """
        The spaCy-based parser parse the sentence into scene graphs based on the dependency parsing
        of the sentence by spaCy.

        All entities (nodes) of the graph come from the noun chunks in the sentence. And the dependencies
        between noun chunks are used for determining the relations among these entities.

        The parsing is performed in three steps:

            1. find all the noun chunks as the entities, and resolve the modifiers on them.
            2. determine the subject of verbs (including nsubj, acl and pobjpass). Please refer to the comments
            in the code for better explanation.
            3. determine all the relations among entities.
            解析主要分为三个步骤：第一步就是找句子的名词和他们的修饰语；第二步就是找到句子中动词的主语；第三步就是找到各个名词之间的关系

        """
        doc = self.nlp(sentence)
        for token in doc:

            if token.dep_ == 'ROOT' and token.pos_ == 'AUX':
                sentence = sentence.replace(str(token), '')
            if token.dep_ == 'nummod' and token.head.pos_ == 'NOUN' and token.pos_ == 'NUM':
                sentence = sentence.replace(str(token), '')
                k = spoken_word_to_number(str(token)) - 1
                for i in range(k):
                    j = sentence.find(str(token.head))
                    sentence = str_insert(sentence, j, str(token.head) + ' and ')
        sentence = sentence.lstrip()
        sentence = sentence.replace('  ', ' ')
        doc = self.nlp(sentence)
        # Step 1: determine the entities.
        entities = list()
        entity_chunks = list()
        for entity in doc.noun_chunks:
            # Ignore pronouns such as "it".
            if entity.root.lemma_ == '-PRON-':
                continue
            ent = dict(
                span=entity.text,
                lemma_span=entity.lemma_,
                head=entity.root.text,
                lemma_head=entity.root.lemma_,
                span_bounds=(entity.start, entity.end),
                modifiers=[]
            )
            for x in entity.root.children:
                # TODO(Jiayuan Mao @ 08/21): try to determine the number.
                if x.dep_ == 'det':  # 限定词
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'nummod':  # 我猜测是数字修饰词
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'amod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'compound':
                    ent['head'] = x.text + ' ' + ent['head']
                    ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']

            if database.is_scene_noun(ent['lemma_head']):
                ent['type'] = 'scene'
            else:
                ent['type'] = 'unknown'

            entities.append(ent)
            entity_chunks.append(entity)

        # Step 2: determine the subject of the verbs.确定动词的主语

        relation_subj = dict()
        for token in doc:
            if token.dep_ == 'nsubj':
                relation_subj[token.head.i] = token.i
                for x in doc:
                    if x.dep_ == 'conj' and x.head == token.head:
                        relation_subj[x.i] = token.i
            elif token.dep_ == 'acl':
                relation_subj[token.i] = token.head.i  #
            elif token.dep_ == 'pobj' and token.head.dep_ == 'agent' and token.head.head.pos_ == 'VERB':
                relation_subj[token.head.head.i] = token.i
        # Step 3: determine the relations.
        relations = list()
        fake_noun_marks = set()
        for entity in doc.noun_chunks:

            # Again, the subjects and the objects are represented by their position.用位置来表示主体，宾语
            relation = None

            # E.g., A woman is [playing] the [piano].
            # E.g., The woman [is] a [pianist].
            if entity.root.dep_ in ('dobj', 'attr') and entity.root.head.i in relation_subj:  # 直接宾语,我猜意思是表语
                relation = {
                    'subject': relation_subj[entity.root.head.i],
                    'object': entity.root.i,
                    'relation': entity.root.head.text,
                    'lemma_relation': entity.root.head.lemma_
                }

            elif entity.root.dep_ == 'pobj':
                # E.g., The piano is played [by] a [woman].
                if entity.root.head.dep_ == 'agent':
                    pass

                # E.g., A [woman] is playing with the piano in the [room].
                elif (
                        entity.root.head.head.pos_ == 'VERB' and
                        entity.root.head.head.i + 1 == entity.root.head.i and
                        database.is_phrasal_verb(entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_)
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }

                # E.g., A [woman] is playing the piano in the [room]. Note that room.head.head == playing.
                # E.g., A [woman] playing the piano in the [room].
                elif (
                        entity.root.head.head.pos_ == 'VERB' or
                        entity.root.head.head.dep_ == 'acl'
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }

                # E.g., A [woman] in front of a [piano].
                elif (
                        entity.root.head.head.dep_ == 'pobj' and
                        database.is_phrasal_prep(str(doc[entity.root.head.head.head.i:entity.root.head.i + 1]).lower())
                ):
                    fake_noun_marks.add(entity.root.head.head.i)
                    if entity.root.head.head.head.head.dep_ == 'acl':
                        relation = {
                            'subject': entity.root.head.head.head.head.head.i,
                            'object': entity.root.i,
                            'relation': doc[entity.root.head.head.head.i:entity.root.head.i + 1].text,
                            'lemma_relation': doc[entity.root.head.head.head.i:entity.root.head.i].lemma_
                        }
                    else:
                        relation = {
                            'subject': entity.root.head.head.head.head.i,
                            'object': entity.root.i,
                            'relation': doc[entity.root.head.head.head.i:entity.root.head.i + 1].text,
                            'lemma_relation': doc[entity.root.head.head.head.i:entity.root.head.i].lemma_
                        }
                # E.g., A [piano] in the [room].

                elif entity.root.head.head.pos_ == 'NOUN':
                    relation = {
                        'subject': entity.root.head.head.i,
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }

                # E.g., A [piano] next to a [woman].
                elif entity.root.head.head.dep_ in ('amod', 'advmod') and entity.root.head.head.head.pos_ == 'NOUN':
                    relation = {
                        'subject': entity.root.head.head.head.i,
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }

                # E.g., A [woman] standing next to a [piano].
                elif entity.root.head.head.dep_ in ('amod',
                                                    'advmod') and entity.root.head.head.head.pos_ == 'VERB' and entity.root.head.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }

                # E.g., A [woman] is playing the [piano] in the room
                elif entity.root.head.head.dep_ == 'VERB' and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }

            # E.g., The [piano] is played by a [woman].
            elif entity.root.dep_ == 'nsubjpass' and entity.root.head.i in relation_subj:
                # Here, we reverse the passive phrase. I.e., subjpass -> obj and objpass -> subj.
                relation = {
                    'subject': relation_subj[entity.root.head.i],
                    'object': entity.root.i,
                    'relation': entity.root.head.text,
                    'lemma_relation': entity.root.head.lemma_
                }

            if relation is not None:

                relations.append(relation)

        # Apply the `fake_noun_marks`.
        entities = [e for e, ec in zip(entities, entity_chunks) if ec.root.i not in fake_noun_marks]
        entity_chunks = [ec for ec in entity_chunks if ec.root.i not in fake_noun_marks]
        filtered_relations = list()

        for token in doc:
            if token.dep_ == 'conj':
                if token.pos_ == 'NOUN':
                    relations_ = copy.deepcopy(relations)
                    for relation in relations_:
                        if token.i == relation['subject']:
                            a = {
                                'subject': token.head.i,
                                'object': relation['object'],
                                'relation': relation['relation'],
                                'lemma_relation': relation['lemma_relation']
                            }
                            relations.append(a)
                        if token.i == relation['object']:
                            a = {
                                'subject': token.head.i,
                                'object': relation['object'],
                                'relation': relation['relation'],
                                'lemma_relation': relation['lemma_relation']
                            }
                            relations.append(a)
                        if token.head.i == relation['subject']:
                            a = {
                                'subject': token.i,
                                'object': relation['object'],
                                'relation': relation['relation'],
                                'lemma_relation': relation['lemma_relation']
                            }
                            relations.append(a)
                        if token.head.i == relation['object']:
                            a = {
                                'subject': token.i,
                                'object': relation['object'],
                                'relation': relation['relation'],
                                'lemma_relation': relation['lemma_relation']
                            }
                            relations.append(a)
        for relation in relations:
            # Use a helper function to map the subj/obj represented by the position
            # back to one of the entity nodes.
            relation['subject'] = self.__locate_noun(entity_chunks, relation['subject'])
            relation['object'] = self.__locate_noun(entity_chunks, relation['object'])
            if relation['subject'] is not None and relation['object'] is not None:
                filtered_relations.append(relation)

        if return_doc:
            return {'entities': entities, 'relations': filtered_relations}, doc
        return {'entities': entities, 'relations': filtered_relations}

    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None
