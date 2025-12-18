def load_raw_text_data(dataset_path: str) -> str:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return raw_text  