import os
import re
import json
from tqdm import tqdm


crypto_class = {
    '3des':['3des','triple-des','des-ede','desede'],
    'aes':['aes','rijndael'],
    'aria':['aria'],
    'blake2':['blake2'],
    'blowfish':['blowfish'],
    'camellia':['camellia'],
    'cast':['cast'],
    'chacha20':['chacha20'],
    'cmac':['cmac'],
    'curve25519':['curve25519','curve-25519','x25519','ristretto255'],
    'curve448':['curve448','curve-448','x448'],
    'des':['des'],
    'dh':['dh','diffie hellman','diffie-hellman','diffiehellman'],
    'dsa':['dsa'],
    'ecc':['ecc',' ec ','elliptic','curve'],
    'ecdh':['ecdh'],
    'ecdsa':['ecdsa'],
    'ecjpake':['ecjpake'],
    'ed448':['ed448','ed-448'],
    'ed25519':['ed25519','ed-25519'],
    'hmac':['hmac'],
    'idea':['idea'],
    'md4':['md4'],
    'md5':['md5'],
    'mdc2':['mdc2','mdc-2'],
    'poly1305':['poly1305','poly-1305'],
    'rc2':['rc2'],
    'rc4':['rc4'],
    'ripemd160':['ripemd160','ripemd-160'],
    'rsa':['rsa'],
    'salsa20':['salsa20'],
    # 'seed':['seed'],
    'sha1':['sha1','sha-1'],
    'sha224':['sha224','sha-224'],
    'sha256':['sha256','sha-256'],
    'sha384':['sha384','sha-384'],
    'sha512':['sha512','sha-512'],
    'sha3':['sha3','keccak'],
    'siphash':['siphash'],
    'sm2':['sm2'],
    'sm3':['sm3'],
    'sm4':['sm4'],
    'tea':['tea'],
    'umac':['umac'],
    'whirlpool':['whirlpool'],
    'xtea':['xtea','x-tea'],
}

block_crypto_mode = [
    "cbc",
    "pcbc",
    "cfb",
    "ctr",
    "ecb",
    "ofb",
    "ocf",
    "xts",
]

AE_mode = [
    "ccm",
    "gcm",
    "sgcm",
    "cwc",
    "eax",
    "ocb",
    "siv",
    "iapm",
]

# add more keyword field if you want

def find_whole_word(word, text):
    pattern = r'\b{}\b'.format(re.escape(word))
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return None


def get_label(name,scode,summary):
    name = name.lower().replace("_"," ")
    scode = scode.lower().replace("_"," ")
    summary = summary.lower().replace("_"," ")
    keywords = dict()
    keywords['crypto_class'] = set()
    # check crypto class:
    for c in crypto_class:
        for keyword in crypto_class[c]:
            res = find_whole_word(keyword, name) # do not modify this line
            if res:
                keywords['crypto_class'].add(c)
                break
            res = find_whole_word(keyword, scode)
            if res:
                keywords['crypto_class'].add(c)
                break
            res = find_whole_word(keyword, summary)
            if res:
                keywords['crypto_class'].add(c)
                break
    
    # check crypto mode:
    keywords['block_mode'] = set()
    for mode in block_crypto_mode:
        # res = find_whole_word(mode, name)
        res = mode if mode in name else None
        if res:
            keywords['block_mode'].add(mode)
            continue
        res = find_whole_word(mode, scode)
        if res:
            keywords['block_mode'].add(mode)
            continue
        res = find_whole_word(mode, summary)
        if res:
            keywords['block_mode'].add(mode)
            continue

    keywords['ae_mode'] = set()
    for mode in AE_mode:
        # res = find_whole_word(mode, name)
        res = mode if mode in name else None
        if res:
            keywords['ae_mode'].add(mode)
            continue
        res = find_whole_word(mode, scode)
        if res:
            keywords['ae_mode'].add(mode)
            continue
        res = find_whole_word(mode, summary)
        if res:
            keywords['ae_mode'].add(mode)
            continue

    return keywords


def get_keywords_from_prediction(name="", summary=""):
    name = name.lower().replace("_"," ")
    summary = summary.lower().replace("_"," ")
    keywords = dict()
    keywords['crypto_class'] = set()
    # check crypto class:
    for c in crypto_class:
        for keyword in crypto_class[c]:
            res = find_whole_word(keyword, name) # do not modify this line
            if res:
                keywords['crypto_class'].add(c)
                break
            res = find_whole_word(keyword, summary)
            if res:
                keywords['crypto_class'].add(c)
                break
    
    # check crypto mode:
    keywords['block_mode'] = set()
    for mode in block_crypto_mode:
        # res = find_whole_word(mode, name)
        res = mode if mode in name else None
        if res:
            keywords['block_mode'].add(mode)
            continue
        res = find_whole_word(mode, summary)
        if res:
            keywords['block_mode'].add(mode)
            continue

    keywords['ae_mode'] = set()
    for mode in AE_mode:
        # res = find_whole_word(mode, name)
        res = mode if mode in name else None
        if res:
            keywords['ae_mode'].add(mode)
            continue
        res = find_whole_word(mode, summary)
        if res:
            keywords['ae_mode'].add(mode)
            continue

    return keywords


def check_label(label_keywords,pred_keywrods):
    results = dict()
    # calc all
    fp,tp = 0,0
    for k in  pred_keywrods['crypto_class']:
        if k in label_keywords['crypto_class']:
            tp+=1
        else:
            fp+=1
    
    for k in pred_keywrods['block_mode']:
        if k in label_keywords['block_mode']:
            tp+=1
        else:
            fp+=1

    for k in pred_keywrods['ae_mode']:
        if k in label_keywords['ae_mode']:
            tp+=1
        else:
            fp+=1
    all_in_label = len(label_keywords['crypto_class'])+len(label_keywords['block_mode'])+len(label_keywords['ae_mode'])
    all_in_pred = len(pred_keywrods['crypto_class'])+len(pred_keywrods['block_mode'])+len(pred_keywrods['ae_mode'])
    if all_in_pred != 0:
        prec = tp/(tp+fp)
    else:
        if all_in_label == 0:
            prec = 1.0
        else:
            prec = 0.0
    if all_in_label != 0:
        recall = tp/all_in_label
    else:
        if all_in_pred == 0:
            recall = 1.0
        else:
            recall = 0.0
    f1 = 0.0 if prec+recall==0 else 2*prec*recall/(prec+recall)
    results['metric_whole'] = {
        'precision':prec,
        'recall':recall,
        'f1':f1
    }
    # only calc crypto_class
    fp,tp = 0,0,
    for k in  pred_keywrods['crypto_class']:
        if k in label_keywords['crypto_class']:
            tp+=1
        else:
            fp+=1
    all_in_label = len(label_keywords['crypto_class'])
    all_in_pred = len(pred_keywrods['crypto_class'])
    if all_in_pred != 0:
        prec = tp/(tp+fp)
    else:
        if all_in_label == 0:
            prec = 1.0
        else:
            prec = 0.0
    if all_in_label != 0:
        recall = tp/all_in_label
    else:
        if all_in_pred == 0:
            recall = 1.0
        else:
            recall = 0.0
    f1 = 0.0 if prec+recall==0 else 2*prec*recall/(prec+recall)
    results['metric_crypto_class'] = {
        'precision':prec,
        'recall':recall,
        'f1':f1
    }
    return results

