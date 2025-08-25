import gc
import torch
from comet import download_model, load_from_checkpoint
from sacrebleu import BLEU
from utils import load_managers
from typing import List
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.generation_metrics import *
from lm_polygraph.estimators import *
import pathlib
import re


MODELS = ['llama', 'gemma', 'eurollm']
DATASETS = [
    'wmt14_csen',
    'wmt14_deen',
    'wmt14_ruen',
    'wmt14_fren',
    'wmt19_deen',
    'wmt19_fien',
    'wmt19_lten',
    'wmt19_ruen',
]


source_ignore_regex = re.compile("(?s).*Original:\n(.*?)\nTranslation:\n")
instruct_source_ignore_regex = re.compile("(?s).*Original: (.*?)<")

translation_ignore_regex = None
instruct_translation_ignore_regex = re.compile("^Translation: ")

torch.set_float32_matmul_precision("medium")

def get_bleu_scores(
    translated_sentences: List[str],
    reference_sentences: List[str],
):
    bleu = BLEU(effective_order=True)
    scores = [bleu.sentence_score(translated_sentences[i], [reference_sentences[i]]).score for i in range(len(translated_sentences))]
    signature = bleu.get_signature()

    return scores, signature


managers = {}
for model in MODELS:
    for model_type in ['base', 'instruct']:
        for split in ['train', 'test']:
            prefix = '' if model_type == 'base' else '_instruct'

            pathlib.Path(f'processed_mans').mkdir(parents=True, exist_ok=True)

            for dataset in DATASETS:
                manager = UEManager.load(f'/workspace/processed_mans/{model}{prefix}_{dataset}_{split}.man')
                managers[f'{model}{prefix}_{dataset}_{split}_full_enriched.man'] = manager

                original_sentences = manager.stats['input_texts']
                translated_sentences = manager.stats['greedy_texts']
                reference_sentences = manager.stats['target_texts']

                manager.gen_metrics[('sequence', 'bleu_proper')] = get_bleu_scores(translated_sentences, reference_sentences)[0]


comet = Comet(source_ignore_regex=source_ignore_regex, translation_ignore_regex=None, gpus=1)
for name, manager in managers.items():
    if 'instruct' in name:
        comet.source_ignore_regex = instruct_source_ignore_regex
        comet.translation_ignore_regex = instruct_translation_ignore_regex
    else:
        comet.source_ignore_regex = source_ignore_regex
        comet.translation_ignore_regex = translation_ignore_regex

    reference_sentences = manager.stats['target_texts']
    manager.gen_metrics[('sequence', str(comet))] = comet(manager.stats, reference_sentences)
    manager.save(f'/workspace/processed_mans/{name}')
del comet
gc.collect()
torch.cuda.empty_cache()


xcomet = XComet(source_ignore_regex=source_ignore_regex, translation_ignore_regex=None, gpus=1)
for name, manager in managers.items():
    if 'instruct' in name:
        xcomet.source_ignore_regex = instruct_source_ignore_regex
        xcomet.translation_ignore_regex = instruct_translation_ignore_regex
    else:
        xcomet.source_ignore_regex = source_ignore_regex
        xcomet.translation_ignore_regex = translation_ignore_regex

    reference_sentences = manager.stats['target_texts']
    manager.gen_metrics[('sequence', str(xcomet))] = xcomet(manager.stats, reference_sentences)
    manager.save(f'/workspace/processed_mans/{name}')
del xcomet
gc.collect()
torch.cuda.empty_cache()


xmetric = XMetric(source_ignore_regex = source_ignore_regex, translation_ignore_regex = None,
                  model_name_or_path="google/metricx-24-hybrid-xxl-v2p6", tokenizer_name="google/mt5-xxl")
for name, manager in managers.items():
    if 'instruct' in name:
        xmetric.source_ignore_regex = instruct_source_ignore_regex
        xmetric.translation_ignore_regex = instruct_translation_ignore_regex
    else:
        xmetric.source_ignore_regex = source_ignore_regex
        xmetric.translation_ignore_regex = translation_ignore_regex

    reference_sentences = manager.stats['target_texts']
    manager.gen_metrics[('sequence', str(xmetric))] = xmetric(manager.stats, reference_sentences)
    manager.save(f'/workspace/processed_mans/{name}')
del xmetric
gc.collect()
torch.cuda.empty_cache()


comet_qe = CometQE(source_ignore_regex=source_ignore_regex, translation_ignore_regex=None, gpus=1, model="Unbabel/wmt23-cometkiwi-da-xxl")
for name, manager in managers.items():
    if 'instruct' in name:
        comet_qe.source_ignore_regex = instruct_source_ignore_regex
        comet_qe.translation_ignore_regex = instruct_translation_ignore_regex
    else:
        comet_qe.source_ignore_regex = source_ignore_regex
        comet_qe.translation_ignore_regex = translation_ignore_regex

    manager.estimations[('sequence', str(comet_qe))] = comet_qe(manager.stats)
    manager.save(f'/workspace/processed_mans/{name}')
del comet_qe
gc.collect()
torch.cuda.empty_cache()


comet_qe = CometQE(source_ignore_regex=source_ignore_regex, translation_ignore_regex=None, gpus=1, model="Unbabel/wmt23-cometkiwi-da-xl")
for name, manager in managers.items():
    if 'instruct' in name:
        comet_qe.source_ignore_regex = instruct_source_ignore_regex
        comet_qe.translation_ignore_regex = instruct_translation_ignore_regex
    else:
        comet_qe.source_ignore_regex = source_ignore_regex
        comet_qe.translation_ignore_regex = translation_ignore_regex

    manager.estimations[('sequence', str(comet_qe))] = comet_qe(manager.stats)
    manager.save(f'/workspace/processed_mans/{name}')
del comet_qe
gc.collect()
torch.cuda.empty_cache()


comet_qe = CometQE(source_ignore_regex=source_ignore_regex, translation_ignore_regex=None, gpus=1, model="Unbabel/wmt22-cometkiwi-da")
for name, manager in managers.items():
    if 'instruct' in name:
        comet_qe.source_ignore_regex = instruct_source_ignore_regex
        comet_qe.translation_ignore_regex = instruct_translation_ignore_regex
    else:
        comet_qe.source_ignore_regex = source_ignore_regex
        comet_qe.translation_ignore_regex = translation_ignore_regex

    manager.estimations[('sequence', str(comet_qe))] = comet_qe(manager.stats)
    manager.save(f'/workspace/processed_mans/{name}')
del comet_qe
gc.collect()
torch.cuda.empty_cache()


for name, manager in managers.items():
    manager.save(f'/workspace/processed_mans/{name}')
