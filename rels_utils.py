# script to generate json files of a user-defined set of linear relations matched with a given symbol
# script to evaluate various linear relations satisfied by the symbols of the 3-point form factor of phi2

import io
import os
import sys
import math
import re
import numpy as np
import time
import datetime
import itertools
from itertools import permutations
from aiamplitudes_common_public import alphabet, quad_prefix
from aiamplitudes_common_public.rels_utils import (first_entry_rel_table,double_adjacency_rel_table,
                                                   triple_adjacency_rel_table,integral_rel_table,get_rel_table_dihedral,
                                                   final_entries_rel_table,initial_entries_rel_table,
                                                   check_rel,check_coeffs_in_rel,get_dihedral_terms_in_symb,get_rel_terms_in_symb,
                                                   get_rel_instances_in_symb,update_rel_instances_in_symb)
import random
import json


##############################################################################################
# GENERATE RELATION INSTANCES #
##############################################################################################
def output_rel_instances_jsons(loop, symb, outpath, rels_to_generate, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Output json files of generated relation instances.
    ---------
    INPUTS:
    loop: int; loop order.
    symb: dict; usually truth symbol, so that this function is independent of any model.
    outpath: str; folder path for output json files
    rels_to_generate: dict; format as the default `rels_to_generate_default`.
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    json files in outpath folder, one for a specific relation named in the format `rel_instances_first0.json`.
    '''
    print('Relation Generation starts at: ', datetime.datetime.now())

    for rel_key, rel_info in rels_to_generate.items():

        if rel_key == 'first':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(first_entry_rel_table[i], rel_slot=0, loop=loop,
                                                           ninstance=rel_info[0][i], symb=symb,
                                                           min_overlap=rel_info[1][i], format=format, seed=seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('First entry relations done at: ', datetime.datetime.now())

        if rel_key == 'initial':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(initial_entries_rel_table[i], rel_slot=0, loop=loop,
                                                           ninstance=rel_info[0][i], symb=symb,
                                                           min_overlap=rel_info[1][i], format=format, seed=seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Multiple-initial entry relations done at: ', datetime.datetime.now())

        if rel_key == 'double':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(double_adjacency_rel_table[i], rel_slot=None, loop=loop,
                                                           ninstance=rel_info[0][i], symb=symb,
                                                           min_overlap=rel_info[1][i], format=format, seed=seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Double adjacency relations done at: ', datetime.datetime.now())

        if rel_key == 'triple':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(triple_adjacency_rel_table[i], rel_slot=None, loop=loop,
                                                           ninstance=rel_info[0][i], symb=symb,
                                                           min_overlap=rel_info[1][i], format=format, seed=seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Triple adjacency relations done at: ', datetime.datetime.now())

        if rel_key == 'integral':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(integral_rel_table[i], rel_slot=None, loop=loop,
                                                           ninstance=rel_info[0][i], symb=symb,
                                                           min_overlap=rel_info[1][i], format=format, seed=seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Integrability relations done at: ', datetime.datetime.now())

        if rel_key == 'final':
            for i in range(len(rel_info[0])):
                rel_instance_list = generate_rel_instances(final_entries_rel_table[i], rel_slot=-1, loop=loop,
                                                           ninstance=rel_info[0][i], symb=symb,
                                                           min_overlap=rel_info[1][i], format=format, seed=seed)
                rel_instance_in_symb_list = get_rel_instances_in_symb(rel_instance_list, symb)

                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_in_symb_list, f)

            print('Final entries relations done at: ', datetime.datetime.now())

        if rel_key == 'dihedral':
            for i in range(len(rel_info[0])):
                rel_instance_list, cycle_instance_list, flip_instance_list = generate_dihedral_rel_instances(loop=loop,
                                                                                                             ninstance=
                                                                                                             rel_info[
                                                                                                                 0][i],
                                                                                                             symb=symb,
                                                                                                             min_overlap=
                                                                                                             rel_info[
                                                                                                                 1][i],
                                                                                                             format=format,
                                                                                                             seed=seed)
                with open(outpath + 'rel_instances_{}{}.json'.format(rel_key, i), 'w') as f:
                    json.dump(rel_instance_list, f)
                with open(outpath + 'rel_instances_{}{}.json'.format("cycle", i), 'w') as f:
                    json.dump(cycle_instance_list, f)
                with open(outpath + 'rel_instances_{}{}.json'.format("flip", i), 'w') as f:
                    json.dump(flip_instance_list, f)
            print('Dihedral relations done at: ', datetime.datetime.now())

    return None

def generate_rel_instances(rel, rel_slot, loop, ninstance, symb=None, min_overlap=0, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Given a number of random relation instances for a given relation.
    If using default, then the generated relation instances are completely independent of any symb.
    If there is input symb and min_overlap >0, then we need at least min_overlap number of words
    in the relation instances to also occur in the given symb; otherwise, reject the generated instances.
    ---------
    INPUTS:
    rel: dict; one specific linear relation in the relation look-up tables.
    rel_slot: int and None; slot in which the relation keywords are to be inserted; between 0 and inf;
              -1 for placing relation keywords at the end, i.e., for final entries relations;
              and None for any slot.
    loop: int; loop order of the generated words.
    ninstance: int; number of relation instances to be generated.
    symb: dict; input symbol to boost efficiency, shoud be truth symbol at a given loop; default None.
    min_overlap: int; min number of overlapping words occuring both in the relation relation and in
                 the input symbol in order to accept the generated instance; default 0; only meaningful
                 if symb is not None.
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    rel_instance_list: list of dicts; each item in the list is a dict corresponding to one relation instance,
                       where the key is a randomly generated word realizing the specific relation instance,
                       the value is the corresponding rel coeff, and the relation keywords are all placed at
                       the pre-specified rel_slot.
    '''

    nterm_rel = len(rel)  # number of terms in the rel
    nletter_rel = len(list(rel.keys())[0])  # number of letters in each term in the rel
    if format == 'full':
        nletter_subword = 2*loop-nletter_rel
    if format == 'quad':
        nletter_subword = 2*loop-4-nletter_rel
    if format == 'oct':
        nletter_subword = 2*loop-8-nletter_rel
    rel_instance_list = []
    max_trial = 1000000
    n_trial = 0
    while len(rel_instance_list) < ninstance:
        n_trial += 1
        if n_trial > max_trial:
            break
        i = n_trial
        subword = generate_random_word(nletter_subword, format='full', seed=seed+nletter_subword+i+1)
        # generate random prefix for compact formats
        if format == 'quad':
            random.seed((seed+100*i)*10)
            prefix = quad_prefix[random.randint(0, len(quad_prefix)-1)]
        if format == 'oct': # not yet implemented
            #random.seed((seed+1000*i)*100)
            #prefix = oct_prefix[random.randint(0, len(oct_prefix)-1)]
            prefix = '1'
        rel_instance = dict()

        for key, value in rel.items():
            if rel_slot == -1: # final entries relations
                if format == 'full':
                    word = subword + key
                else: # no final entries relations for compact formats
                    #word = None
                    return None

            elif rel_slot == None:
                random.seed((seed + nletter_subword + i + 1) * 100)
                rel_slot_any = random.randint(0, 2 * loop)
                if format == 'full':
                    word = subword[:rel_slot_any] + key + subword[rel_slot_any:]
                else:  # compact formats
                    word = prefix + subword[:rel_slot_any] + key + subword[rel_slot_any:]

            else:
                if format == 'full':
                    word = subword[:rel_slot] + key + subword[rel_slot:]
                else: # compact formats
                    word = prefix + subword[:rel_slot] + key + subword[rel_slot:]

            rel_instance.update({word: value})

        if symb:
            count = 0
            for word in list(rel_instance.keys()):
                if word in symb:
                    count += 1
            if count >= min_overlap:
                rel_instance_list.append(rel_instance)

        else:
            rel_instance_list.append(rel_instance)

    return rel_instance_list

def generate_dihedral_rel_instances(loop, ninstance, symb=None, min_overlap=0, format={'full'}, seed=0):
    '''
    '''
    #nterm_rel = len(rel)  # number of terms in the rel
    #nletter_rel = len(list(rel.keys())[0])  # number of letters in each term in the rel
    if format == 'full':
        nletter_subword = 2*loop
    dihedral_instance_list = []
    cycle_instance_list = []
    flip_instance_list = []

    max_trial = 1000000
    n_trial = 0
    seen_words_dihedral = {}

    while (len(cycle_instance_list) < ninstance) or (len(flip_instance_list) < ninstance):
        # pull dihedral relations directly from the symb.
        #generate a random word from the symbol and get images
        key, coef = random.choice(list(symb.items()))
        n_trial += 1
        if n_trial > max_trial:
            break
        if key in seen_words_dihedral: continue
        images_coeffs = get_dihedral_terms_in_symb(key,symb)
        #add all images to the hash table so we don't generate the same rel twice
        seen_words_dihedral = seen_words_dihedral | images_coeffs
        dihedral_instance_list,cycle_instance_list,flip_instance_list = (
            get_relpairs(images_coeffs,dihedral_instance_list,cycle_instance_list,flip_instance_list))

    return dihedral_instance_list, cycle_instance_list, flip_instance_list

def get_relpairs(images_set,dihedral_instance_list=[],cycle_instance_list=[],flip_instance_list=[]):
    '''
    Given a dict of images, get all pairs and make two-term relations. Return the set in a list given as an arg.
    ---------
    INPUTS:
    images_set: dict, set of dihedral images of a key
    rel_instance_list: the list of relation instances to append the pairs to
    OUTPUTS:
    rel_instance_list: the list of relation instances to append the pairs to, with pairs appended
    '''
    cycle1_set={0,3,4}
    cycle2_set={1,2,5}
    my_cycle_list=[]
    my_flip_list=[]
    # get all possible pairs of dihedral images, merge each pair into one two-term dict
    for a_idx, a in enumerate(list(images_set.keys())):
        for my_idx, b in enumerate(list(images_set.keys())[a_idx + 1:]):
            b_idx=my_idx+a_idx+1
            pair = {a: images_set[a], b: images_set[b]}
            #update the rel coeffs
            for idx, (word, symb_coeff) in enumerate(pair.items()):
                if idx == 0:
                    rel_coeff = 1
                else:
                    rel_coeff = -1
                pair.update({word: [symb_coeff, rel_coeff]})
            #one dict with all pairs
            dihedral_instance_list.append(pair)
            #if the pair is in a cycle or a flip
            if (a_idx in cycle1_set and b_idx in cycle1_set) or (a_idx in cycle2_set and b_idx in cycle2_set):
                my_cycle_list.append(pair)
            else:
                my_flip_list.append(pair)

    # randomly pull one instance from the set. overlaps lead to noisy curves, since we're double-counting words
    mynum = random.random()
    if mynum > 0.5:
        inst = random.choice(my_cycle_list)
        cycle_instance_list.append(inst)
        dihedral_instance_list.append(inst)
    else:
        inst = random.choice(my_flip_list)
        flip_instance_list.append(inst)
        dihedral_instance_list.append(inst)

    return dihedral_instance_list, cycle_instance_list, flip_instance_list
def generate_random_word(word_length, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Generate a random word with a specific length.
    ---------
    INPUTS:
    word_length: int; number of letters in the generated word.
    seed: int; random number generating seed; default 0.

    OUTPUTS:
    word: str; a word with letters in the alphabet and the specific length.
    '''
    word = ''
    for i in range(word_length):
        random.seed(seed + i)
        word += alphabet[random.randint(0, len(alphabet) - 1)]
    if format == 'full':
        return word
    if format == 'quad':
        print(word,quad_prefix)
        random.seed((seed+100)*10) # must generate another random number
        prefix = quad_prefix[random.randint(0, len(quad_prefix)-1)]
        return prefix+word
    if format == 'oct': # not yet implemented
        #random.seed((seed+1000)*100) # must generate another random number
        #prefix = oct_prefix[random.randint(0, len(quad_prefix)-1)]
        #return prefix+word
        return None


##############################################################################################
# WORD-ORIENTED APPROACH (works at 5 loops, but not scalable!) #
##############################################################################################


# The default relations to check in the format of {rel name: [frac, ..., frac]},
# where frac (0-1) is the fraction of the total words in a symbol to be checked
# and the list order follows rel ID as defined in the relation tables.
# If some rel ID is not to be checked, then set its corresponding frac to 0.
# Can be modified according to the user's needs.

rels_to_check_default = {'first': [0.1, 0.1, 0.1], 'double': [0.1, 0.1, 0.1], 'triple': [0.1],
                         'final': [0.1]*29, 'integral': [0.01, 0.01, 0.01]}


def assess_rels_viaword(symb, symb_truth, rels_to_check, p_norm=None, format={'full', 'quad', 'oct'}, seed=0):
    '''
    Assess the success rate of a set of user-specified relations for a given symbol (usually model predictions),
    thru rels_to_check for word-oriented approach.
    One symbol only, which means one epoch model output.
    ---------
    INPUTS:
    symb: dict; usually model predictions.
    symb_truth: dict; truth symbol.
    rels_to_check: dict; user-defined relations to check; formats same as the two defaults defined above.
    p_norm: float or None; p is the average accuracy of model prediction;
            goal is to normalize the accuracy by the number of terms in a relation; default None.
    format: string; different formats to represent the words;
            only accept three choices---'full', 'quad', 'oct';
            if format is 'full', then can use `rels_to_check_full_default`;
            if format is 'quad' or 'oct, then can use `rels_to_check_compact_default`;
            note that here relations living across the seam are not considered, i.e.,
            only relations among exposed letters are considered; the format prefix is generated randomly.
    seed: int; random number generating seed; default 0;
          in principle, can even have a seed_list for different relations (currently not implemented).

    OUTPUTS:
    rel_acc: dict; format: {rel ID: [percent, percent_allcorrect, percent_magcorrect, percent_signcorrect, num_rels]},
                    where percent is the percent of correct relations,
                    percent_allcorrect is the percent of correct relations with all its coeffs correct,
                    percent_magcorrect is the percent of correct relations with all its coeffs mag-correct,
                    percent_signcorrect is the percent of correct relations with all its coeffs sign-correct,
                    and num_rels is the number of relation instances.
    '''
    print('Assessment via word approach starts at: ', datetime.datetime.now())
    rel_acc = dict()

    for rel_key, rel_frac_list in rels_to_check.items():

        if rel_key == 'first':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], first_entry_rel_table[i],
                                                              rel_slot='first', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'first{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                      percent_signcorrect, len(rel_term_in_symb_list)]})
            print('First entry relations done at: ', datetime.datetime.now())

        if rel_key == 'double':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], double_adjacency_rel_table[i],
                                                              rel_slot='any', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'double{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                       percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Double adjacency relations done at: ', datetime.datetime.now())

        if rel_key == 'triple':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], triple_adjacency_rel_table[i],
                                                              rel_slot='any', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'triple{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                       percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Triple adjacency relations done at: ', datetime.datetime.now())

        if rel_key == 'final':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], final_entries_rel_table[i],
                                                              rel_slot='final', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'final{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                      percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Final entries relations done at: ', datetime.datetime.now())

        if rel_key == 'integral':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], integral_rel_table[i],
                                                              rel_slot='any', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'integral{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                         percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Integrability relations done at: ', datetime.datetime.now())

        if rel_key == 'initial':
            for i in range(len(rel_frac_list)):
                rel_term_in_symb_list = get_rel_terms_in_symb(symb, rel_frac_list[i], initial_entries_rel_table[i],
                                                              rel_slot='initial', format=format, seed=seed)
                percent = check_rel(rel_term_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_term_in_symb_list,
                                                                                                  symb_truth,
                                                                                                  return_counts=False)
                rel_acc.update({'initial{}'.format(i): [percent, percent_allcorrect, percent_magcorrect,
                                                      percent_signcorrect, len(rel_term_in_symb_list)]})
            print('Initial entries relations done at: ', datetime.datetime.now())
    return rel_acc

##############################################################################################
# RELATION-ORIENTED APPROACH #
##############################################################################################


def assess_rels_viarel(symb, symb_truth, inpath, p_norm=None):
    '''
    Assess the success rate of a set of user-specified relations for a given symbol (usually model predictions),
    thru the stored relation instances for relation-oriented approach.
    One symbol only, which means one epoch model output.
    ---------
    INPUTS:
    symb: dict; usually model predictions.
    symb_truth: dict; truth symbol.
    inpath: str; folder path for rel_data.
    p_norm: float or None; p is the average accuracy of model prediction;
            goal is to normalize the accuracy by the number of terms in a relation; default None.

    OUTPUTS:
    rel_acc: dict; format: {rel ID: [percent, percent_allcorrect, percent_magcorrect, percent_signcorrect, num_rels]},
                    where percent is the percent of correct relations,
                    percent_allcorrect is the percent of correct relations with all its coeffs correct,
                    percent_magcorrect is the percent of correct relations with all its coeffs mag-correct,
                    percent_signcorrect is the percent of correct relations with all its coeffs sign-correct,
                    and num_rels is the number of relation instances.
    '''
    print('Assessment via relation approach starts at: ', datetime.datetime.now())
    rel_acc = dict()

    for filename in os.listdir(inpath):
        print('Assessing relation file {} ...'.format(filename))
        parts = filename.split("_")
        rel_key = parts[-1].split(".")[0]

        file_path = os.path.join(inpath, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            with open(file_path, 'r') as openfile:
                rel_instance_list = json.load(openfile)

                rel_instance_in_symb_list = update_rel_instances_in_symb(rel_instance_list, symb)
                percent = check_rel(rel_instance_in_symb_list, return_rel_info=False, p_norm=p_norm)
                percent_allcorrect, percent_magcorrect, percent_signcorrect = check_coeffs_in_rel(rel_instance_in_symb_list, symb_truth, return_counts=False)
                rel_acc.update({rel_key: [percent, percent_allcorrect, percent_magcorrect, percent_signcorrect, len(rel_instance_in_symb_list)]})

    print('Assessment via relation approach ends at: ', datetime.datetime.now())
    return rel_acc

def generate_trivial0_symb(nterms_pertype, loop, seed=0, format="full"):
    '''
    Generate a symbol in full format consisting solely of trivial zero terms.
    ---------
    INPUTS:
    nterms_pertype: int; number of terms per type of trivial zero;
                         the total terms in the generated symbol is nterms_pertype * n_types.
    loop: int; loop order of the generated words.
    seed: int; random number generating seed; default 0.
    OUTPUTS:
    trivial0_symb: dict; the generated symbol consisting solely of trivial zero terms.
    '''
    trivial0_symb = dict()

    for rel in first_entry_rel_table:  # prefix rule
        prefix_rels = generate_rel_instances(rel, 0, loop, nterms_pertype, format=format, seed=seed)
        for prefix_rel in prefix_rels:
            for word in prefix_rel.keys():
                trivial0_symb.update({word: 0})

    if format == "full":
        for rel in final_entries_rel_table[:3]:  # suffix rule
            suffix_rels = generate_rel_instances(rel, -1, loop, nterms_pertype, format=format, seed=seed + 1)
            for suffix_rel in suffix_rels:
                for word in suffix_rel.keys():
                    trivial0_symb.update({word: 0})

    for rel in get_rel_table_dihedral(double_adjacency_rel_table):  # adjacency rule
        adjacency_rels = generate_rel_instances(rel, None, loop, nterms_pertype, format=format, seed=seed + 2)
        for adjacency_rel in adjacency_rels:
            for word in adjacency_rel.keys():
                trivial0_symb.update({word: 0})

    return trivial0_symb
