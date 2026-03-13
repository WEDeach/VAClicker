from .shared import state


def update_text_mapping(lang: str, value: dict):
    state.text_mapping.setdefault(lang, {})
    state.text_mapping[lang].update(value)


def get_text_mapping(key: str) -> str:
    lang = state.lang
    r = state.text_mapping.get(lang, {}).get(key)
    if r:
        return r
    raise KeyError(f"Text mapping not found for key: {key} (lang: {lang})")
