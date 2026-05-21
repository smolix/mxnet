# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import io
import importlib.util
import sys
import types
from pathlib import Path


def _load_notebook_test_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / 'tests' / 'utils' / 'notebook_test' / '__init__.py'
    spec = importlib.util.spec_from_file_location('notebook_test_for_unit', str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_notebook_closes_output_scan_file(monkeypatch, tmp_path):
    nbconvert = types.ModuleType('nbconvert')
    preprocessors = types.ModuleType('nbconvert.preprocessors')
    preprocessors.ExecutePreprocessor = object
    nbconvert.preprocessors = preprocessors
    nbformat = types.ModuleType('nbformat')
    nbformat.read = None
    nbformat.write = None
    monkeypatch.setitem(sys.modules, 'nbconvert', nbconvert)
    monkeypatch.setitem(sys.modules, 'nbconvert.preprocessors', preprocessors)
    monkeypatch.setitem(sys.modules, 'nbformat', nbformat)

    module = _load_notebook_test_module()
    scanned_files = []

    class FakeOutput(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    class FakeExecutePreprocessor:
        def __init__(self, **kwargs):
            pass

        def preprocess(self, notebook, resources):
            return notebook, resources

    def fake_read(path, as_version):
        return {'cells': []}

    def fake_write(notebook, output_file):
        Path(output_file).write_text('Warning: generated warning\n', encoding='utf-8')

    def fake_open(path, mode='r', encoding=None):
        output = FakeOutput(Path(path).read_text(encoding=encoding))
        scanned_files.append(output)
        return output

    monkeypatch.setattr(module, 'ExecutePreprocessor', FakeExecutePreprocessor)
    monkeypatch.setattr(module.nbformat, 'read', fake_read)
    monkeypatch.setattr(module.nbformat, 'write', fake_write)
    monkeypatch.setattr(module, 'open', fake_open, raising=False)

    assert not module.run_notebook('example', str(tmp_path), temp_dir=str(tmp_path / 'work'))
    assert len(scanned_files) == 1
    assert scanned_files[0].closed
