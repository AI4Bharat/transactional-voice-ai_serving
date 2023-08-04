import yaml
import os

BASE_HOTWORD_PATH = "hotwords/{}.txt"
ENTITY_VARIATIONS_PATH = "entity-variations/{}.yaml"


def hotword_process(word):
    word = word.strip().lower()
    return word


def get_entity_variations(lang, base_path):
    with open(os.path.join(base_path, ENTITY_VARIATIONS_PATH.format(lang)), encoding="utf-8") as f:
        entities = yaml.load(f, yaml.BaseLoader)
    variations = list()
    for ent_type, variation_dict in entities.items():
        for ent_value, variation_list in variation_dict.items():
            variations.extend(variation_list)
    variations = list(map(lambda x: x.lower(), variations))
    return variations


def get_base_hotwords(lang, base_path):
    with open(os.path.join(base_path, BASE_HOTWORD_PATH.format(lang)), encoding="utf-8") as f:
        hotwords = f.read().splitlines()
    hotwords = sorted(list(map(hotword_process, hotwords)))
    return hotwords


def get_entity_unique_hotwords(lang, base_path):
    base_hw = get_base_hotwords(lang, base_path)
    entity_variations = get_entity_variations(lang, base_path)
    unique_words = list(set(" ".join(entity_variations).split()))
    unique_words = sorted([w for w in unique_words if len(w) > 3])
    hotwords = base_hw + unique_words
    return hotwords


def get_entity_whole_hotwords(lang, base_path):
    base_hw = get_base_hotwords(lang)
    entity_variations = get_entity_variations(lang, base_path)
    entity_variations = sorted(list(set([w for w in entity_variations])))
    hotwords = base_hw + entity_variations
    return hotwords


hotword_to_fn = {
    "base": get_base_hotwords,
    "entities-unique": get_entity_unique_hotwords,
    "entities-whole": get_entity_whole_hotwords,
}
