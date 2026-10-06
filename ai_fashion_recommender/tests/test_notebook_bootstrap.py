"""Check real Notebook setup cells without loading models or training data."""
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['deepfashion_evaluation.ipynb', 'train_fashion_attribute_heads.ipynb']


def setup_cell(name, data_dir, project_input=''):
    notebook = json.loads((PROJECT / name).read_text(encoding='utf-8'))
    cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
    code = ''.join(cell['source'])
    code = code.replace("PROJECT_DIR_INPUT = r''", f'PROJECT_DIR_INPUT = {str(project_input)!r}')
    code = code.replace("DEEPFASHION_ROOT_INPUT = r''", f'DEEPFASHION_ROOT_INPUT = {str(data_dir)!r}')
    return code


@pytest.mark.parametrize('name', NOTEBOOKS)
@pytest.mark.parametrize('cwd', [PROJECT.parent, PROJECT])
def test_setup_finds_project_from_repo_or_notebook_folder(name, cwd, tmp_path, monkeypatch):
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    namespace = {}
    code = setup_cell(name, tmp_path)
    exec(code, namespace)
    assert namespace['PROJECT_DIR'] == PROJECT
    assert sys.path[0] == str(PROJECT / 'src')
    exec(code, namespace)
    assert sys.path.count(str(PROJECT / 'src')) == 1


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_explicit_project_path_works_from_another_folder(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    namespace = {}
    exec(setup_cell(name, tmp_path, PROJECT), namespace)
    assert namespace['PROJECT_DIR'] == PROJECT


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_invalid_project_path_fails_before_dataset_or_model_work(name, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    with pytest.raises(FileNotFoundError):
        exec(setup_cell(name, tmp_path, tmp_path), {})
