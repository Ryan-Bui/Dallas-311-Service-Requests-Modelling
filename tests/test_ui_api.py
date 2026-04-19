import pytest
import shutil
import time
from pathlib import Path
from http import HTTPStatus
from flask import json
import ui.flask_app as flask_app_module
from ui.flask_app import app, UPLOAD_DIR

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Keep the in-memory Flask pipeline state isolated across tests."""
    with flask_app_module._lock:
        flask_app_module._state.update({
            'status': 'idle',
            'progress': 0,
            'logs': [],
            'results': None,
            'error': None,
            'data_path': None,
            'started_at': None,
            'finished_at': None,
        })
        for name in flask_app_module._state['agents']:
            flask_app_module._state['agents'][name] = 'idle'
    yield

@pytest.fixture(autouse=True)
def cleanup_uploads():
    """Ensure upload directory is clean before/after tests."""
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield

def wait_for_idle(client, timeout=5):
    """
    Poll /api/status until the pipeline is no longer 'running'.
    Per documentation: 'Fetch the current resource state before retrying.'
    """
    start = time.time()
    while time.time() - start < timeout:
        res = client.get('/api/status')
        status = res.get_json().get('status')
        if status != 'running':
            return status
        time.sleep(0.1)
    return 'timeout'

def test_config_endpoint(client):
    """Verify that /api/config returns the expected structure."""
    res = client.get('/api/config')
    assert res.status_code == HTTPStatus.OK
    data = res.get_json()
    assert 'default_data_path' in data
    assert 'sample_path' in data
    assert 'available_datasets' in data
    assert isinstance(data['available_datasets'], list)
    if data['sample_path']:
        assert any(item['path'] == data['sample_path'] for item in data['available_datasets'])

def test_upload_endpoint(client, tmp_path):
    """Verify that /api/upload accepts a CSV and saves it."""
    test_csv = tmp_path / "test_upload.csv"
    test_csv.write_text("col1,col2\n1,2")
    
    with open(test_csv, 'rb') as f:
        res = client.post(
            '/api/upload',
            data={'file': (f, 'test_upload.csv')},
            content_type='multipart/form-data'
        )
    
    assert res.status_code == HTTPStatus.OK
    data = res.get_json()
    assert 'path' in data
    assert data['filename'] == 'test_upload.csv'

def test_config_lists_uploaded_files(client, tmp_path):
    """Verify that uploaded CSV files appear in the config dataset list."""
    test_csv = tmp_path / "listed_upload.csv"
    test_csv.write_text("col1,col2\n1,2")

    with open(test_csv, 'rb') as f:
        upload_res = client.post(
            '/api/upload',
            data={'file': (f, 'listed_upload.csv')},
            content_type='multipart/form-data'
        )

    uploaded_path = upload_res.get_json()['path']
    config_res = client.get('/api/config')
    datasets = config_res.get_json()['available_datasets']

    assert any(item['path'] == uploaded_path for item in datasets)

def test_run_endpoint_accepted(client, monkeypatch):
    """Verify that /api/run returns 202 Accepted for background work."""
    class DummyThread:
        def __init__(self, target, args=(), daemon=None, name=None):
            self.target = target
            self.args = args

        def start(self):
            with flask_app_module._lock:
                flask_app_module._state['status'] = 'running'

    monkeypatch.setattr(flask_app_module.threading, 'Thread', DummyThread)
    res = client.post('/api/run', data=json.dumps({}), content_type='application/json')
    assert res.status_code == HTTPStatus.ACCEPTED
    assert res.get_json()['status'] == 'started'

def test_run_endpoint_rejects_missing_file(client):
    """Verify that /api/run fails fast when the requested CSV does not exist."""
    missing_path = Path(UPLOAD_DIR / "missing.csv")
    res = client.post(
        '/api/run',
        data=json.dumps({'data_path': str(missing_path)}),
        content_type='application/json',
    )

    assert res.status_code == HTTPStatus.BAD_REQUEST
    assert "Dataset not found" in res.get_json()['error']

def test_run_endpoint_uses_uploaded_file_path(client, tmp_path, monkeypatch):
    """Verify that the uploaded CSV path is passed into the background runner."""
    test_csv = tmp_path / "new_upload.csv"
    test_csv.write_text("col1,col2\n1,2")

    with open(test_csv, 'rb') as f:
        upload_res = client.post(
            '/api/upload',
            data={'file': (f, 'new_upload.csv')},
            content_type='multipart/form-data'
        )

    uploaded_path = upload_res.get_json()['path']
    captured = {}

    class DummyThread:
        def __init__(self, target, args=(), daemon=None, name=None):
            captured['target'] = target
            captured['args'] = args
            captured['daemon'] = daemon
            captured['name'] = name

        def start(self):
            captured['started'] = True

    monkeypatch.setattr(flask_app_module.threading, 'Thread', DummyThread)

    res = client.post(
        '/api/run',
        data=json.dumps({'data_path': uploaded_path}),
        content_type='application/json',
    )

    assert res.status_code == HTTPStatus.ACCEPTED
    assert res.get_json()['data_path'] == uploaded_path
    assert captured['args'] == (uploaded_path,)
    assert captured['started'] is True

def test_conflict_on_concurrent_run(client, monkeypatch):
    """
    Verify 409 Conflict when attempting to run while already running.
    Follows: '409 Conflict means the request conflicts with the current resource state.'
    """
    class DummyThread:
        def __init__(self, target, args=(), daemon=None, name=None):
            self.target = target
            self.args = args

        def start(self):
            with flask_app_module._lock:
                flask_app_module._state['status'] = 'running'

    monkeypatch.setattr(flask_app_module.threading, 'Thread', DummyThread)

    # Start first run
    client.post('/api/run')
    
    # Try second run immediately
    res = client.post('/api/run', data=json.dumps({}), content_type='application/json')
    
    assert res.status_code == HTTPStatus.CONFLICT
    data = res.get_json()
    assert "Conflict" in data['error']
    assert data['current_status'] == 'running'

def test_reset_endpoint_robust(client):
    """
    Verify that /api/reset clears the state after a completed run.
    """
    with flask_app_module._lock:
        flask_app_module._state['status'] = 'done'
        flask_app_module._state['results'] = {'ok': True}
    
    res = client.post('/api/reset')
    assert res.status_code == HTTPStatus.OK
    
    status_now = client.get('/api/status').get_json()['status']
    assert status_now == 'idle'

def test_reset_conflict_when_running(client):
    """Verify that resetting while running yields 409."""
    with flask_app_module._lock:
        flask_app_module._state['status'] = 'running'
    res = client.post('/api/reset')
    assert res.status_code == HTTPStatus.CONFLICT
    assert "Cannot reset" in res.get_json()['error']
