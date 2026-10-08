"""Collect declared submission files against the immutable task base commit."""
from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath


def declared_paths(messages: list[dict]) -> list[str] | None:
    """Use the last explicit git-diff export; never infer paths from test outcomes."""
    selected = None
    for message in messages:
        for action in message.get('extra', {}).get('actions', []):
            command = action.get('command', '')
            try:
                lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>')
                lexer.whitespace_split = True
                tokens = list(lexer)
            except ValueError:
                continue
            for i in range(len(tokens) - 2):
                if tokens[i:i + 2] == ['git', 'diff']:
                    tail = tokens[i + 2:]
                elif tokens[i:i + 3] == ['git', '--no-pager', 'diff']:
                    tail = tokens[i + 3:]
                else:
                    continue
                end = next((j for j, t in enumerate(tail) if t in ('>', '>>', ';', '&&', '||', '|')), len(tail))
                if end == len(tail) or tail[end] != '>':
                    continue
                args = tail[:end]
                if '--' in args:
                    paths = args[args.index('--') + 1:]
                else:
                    paths = [a for a in args if a not in ('HEAD', '--staged', '--cached', '--binary') and not a.startswith('--unified=')]
                selected = None
                if paths and all(valid_path(p) for p in paths):
                    selected = paths
    return selected


def valid_path(path: str) -> bool:
    return bool(path) and not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts and not any(c in path for c in '*?[$`\n') and not path.startswith('-')


def collection_command(base_commit: str, paths: list[str]) -> str:
    if not re.fullmatch(r'[0-9a-f]{40,64}', base_commit):
        raise ValueError('Expected immutable base commit hash')
    if not paths or not all(valid_path(p) for p in paths):
        raise ValueError('Expected explicit repository-relative paths')
    # A private index includes new files without altering the agent's real index.
    quoted = ' '.join(shlex.quote(p) for p in paths)
    return (
        'set -eu\n'
        'rafd_index=$(mktemp /tmp/rafd-index.XXXXXX)\n'
        'rm -f "$rafd_index"\n'
        'trap \'rm -f "$rafd_index"\' EXIT\n'
        'export GIT_INDEX_FILE="$rafd_index"\n'
        f'git read-tree {base_commit}\n'
        f'git add -A -- {quoted}\n'
        f'git -c core.quotePath=false diff --cached --binary --no-ext-diff --no-textconv {base_commit} -- {quoted}\n'
    )


def collect_submission(env, *, base_commit: str, messages: list[dict], raw: str, directory: Path) -> tuple[str, dict]:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'agent_submission.patch').write_text(raw)
    paths = declared_paths(messages)
    metadata = {'policy': 'declared-files-base-commit-v1', 'base_commit': base_commit, 'paths': paths, 'raw_chars': len(raw)}
    if paths is None:
        metadata.update(status='unverified_scope', changed=False)
        result = raw
    else:
        response = env.execute({'command': collection_command(base_commit, paths)}, cwd='/testbed', timeout=120)
        (directory / 'patch_collection.log').write_text(response.get('output', ''))
        if response['returncode'] != 0:
            metadata.update(status='collection_failed', changed=False, returncode=response['returncode'])
            result = raw
        else:
            result = response['output']
            metadata.update(status='collected', changed=result != raw, collected_chars=len(result))
            (directory / 'collected_submission.patch').write_text(result)
    (directory / 'patch_collection.json').write_text(json.dumps(metadata, indent=2))
    return result, metadata
