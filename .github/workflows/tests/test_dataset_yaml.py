import pathlib, yaml, pytest

def test_dataset_yaml_consistency():
    p = pathlib.Path('dataset.yaml')
    if not p.exists():
        pytest.skip("dataset.yaml não está no root do repositório")
    data = yaml.safe_load(p.read_text())

    for k in ('path','train','val','nc','names'):
        assert k in data, f"'{k}' em falta no dataset.yaml"

    names = data['names']
    if isinstance(names, dict):
        # ordena por id numérico/str
        try:
            ids = sorted(names, key=lambda x: int(x))
        except Exception:
            ids = sorted(names)
        names_list = [names[k] for k in ids]
    else:
        names_list = list(names)
    assert len(names_list) == int(data['nc']), "nc deve bater certo com o nº de classes"