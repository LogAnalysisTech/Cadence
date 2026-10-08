import subprocess

from monitor_runtime.evaluation.patch_collection import collection_command, declared_paths


def test_collect_staged_unstaged_committed_new_deleted_without_mutating_index(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True)
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    for name in ['a.py', 'b.py', 'deleted.py']:
        (tmp_path / name).write_text('original\n')
    git('add', '.')
    git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD').strip()
    (tmp_path / 'a.py').write_text('staged\n')
    git('add', 'a.py')
    git('commit', '-qm', 'agent commit')
    (tmp_path / 'a.py').write_text('later staged\n')
    git('add', 'a.py')
    (tmp_path / 'b.py').write_text('unstaged\n')
    (tmp_path / 'new.py').write_text('new source\n')
    (tmp_path / 'patch.txt').write_text('must not be included')
    (tmp_path / 'deleted.py').unlink()
    before = (tmp_path / '.git/index').read_bytes()
    patch = subprocess.check_output(['bash', '-c', collection_command(base, ['a.py', 'b.py', 'new.py', 'deleted.py'])], cwd=tmp_path, text=True)
    assert (tmp_path / '.git/index').read_bytes() == before
    assert '+later staged' in patch and '+unstaged' in patch and '+new source' in patch
    assert 'deleted file mode' in patch and 'patch.txt' not in patch
    git('reset', '--hard', base)
    (tmp_path / 'new.py').unlink()
    subprocess.run(['git', 'apply', '-'], input=patch, text=True, cwd=tmp_path, check=True)
    assert (tmp_path / 'a.py').read_text() == 'later staged\n'
    assert (tmp_path / 'new.py').read_text() == 'new source\n'
    assert not (tmp_path / 'deleted.py').exists()


def test_export_scope_uses_last_explicit_paths_and_rejects_unsafe_scope():
    def msg(command):
        return {'extra': {'actions': [{'command': command}]}}
    assert declared_paths([msg('git diff --staged -- a.py > patch.txt'), msg('git diff -- a.py b.py > patch.txt; echo done')]) == ['a.py', 'b.py']
    assert declared_paths([msg('git diff -- ../outside > patch.txt')]) is None
    assert declared_paths([msg('git diff -- a.py | head')]) is None

    assert declared_paths([msg("git --no-pager diff -- a.py > patch.txt")]) == ["a.py"]
    assert declared_paths([msg("git diff --staged a.py > patch.txt")]) == ["a.py"]
