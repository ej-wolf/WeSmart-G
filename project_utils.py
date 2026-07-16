import re
from pathlib import Path


def strip_split_suffix(tag: str) -> str:
    """Remove one trailing `_train` or `_test` split suffix."""
    for suffix in ('_train', '_test'):
        if tag.endswith(suffix):
            return tag[:-len(suffix)]
    return tag


def strip_timestamp_prefix(tag: str) -> str:
    """ Remove one `YYMMDD_HH-MM-SS_` run prefix when present."""
    return tag.split('_', 2)[2] if re.match(r'^\d{6}_\d{2}-\d{2}-\d{2}_.+', tag) else tag


def _parse_specs(tag: str) -> tuple[str | None, str | None]:
    """Extract canonical `ft.._w..-..` specs and the raw ft token."""
    ft_match = re.search(r'(?:^|_)(?:ft(?P<ft1>\d+)|(?P<ft2>\d+)ft)(?:_|$)', tag)
    ft = (ft_match.group('ft1') or ft_match.group('ft2')) if ft_match else None

    ws_match = re.search(r'(?:^|_)(?:w(?P<w1>\d+)-(?P<s1>\d+)|(?P<w2>\d+(?:o\d+)?)w-(?P<s2>\d+(?:o\d+)?))(?:_|$)', tag)
    if ws_match:
        win = ws_match.group('w1') or ws_match.group('w2')
        stride = ws_match.group('s1') or ws_match.group('s2')
    else:
        win = stride = None

    parts = []
    if ft is not None:
        parts.append(f"ft{ft}")
    if win is not None and stride is not None:
        parts.append(f"w{win}-{stride}")
    return ('_'.join(parts) if parts else None), ft_match.group(0).strip('_') if ft_match else None


def _strip_same_specs(tag: str, specs: str | None) -> str:
    """Remove one matching spec token from a tag and normalize separators."""
    if not specs:
        return tag
    tag = re.sub(rf'(?:^|_){re.escape(specs)}(?:_|$)', '_', tag)
    tag = re.sub(r'__+', '_', tag).strip('_')
    return tag


def _resolve_epoch_tag(path: Path) -> str | None:
    """Resolve one compact epoch tag: `BM` for best-model refs, else `epNNN`."""
    if path.suffix == '.pt':
        match = re.fullmatch(r'best_model\.(\d+)', path.stem)
        if match:
            return 'BM'
        match = re.fullmatch(r'checkpoint_ep-(\d+)', path.stem)
        if match:
            return f"ep{int(match.group(1))}"
        return None

    if path.exists() and path.is_dir():
        best_models = sorted(path.glob('best_model.*.pt'))
        if best_models:
            return 'BM'
    return None


def resolve_best_pt_model(model_ref) -> Path:
    """ Resolve one model ref to the concrete checkpoint file to load."""
    path = Path(model_ref)
    if path.is_file():
        if path.suffix != '.pt':
            raise FileNotFoundError(f'Expected a .pt model file, got: {path}')
        return path

    if path.exists() and path.is_dir():
        best_models = sorted(path.glob('best_model.*.pt'))
        if best_models:
            return best_models[-1]

        model_pt = path/'model.pt'
        if model_pt.is_file():
            return model_pt

        checkpoints = sorted(path.glob('checkpoint_ep-*.pt'))
        if checkpoints:
            return checkpoints[-1]

        raise FileNotFoundError(f'No model checkpoint found in {path}')

    raise FileNotFoundError(path)


def _resolve_model_parts(value) -> tuple[str, str | None]:
    """Resolve one model tag and optional epoch tag from a path or plain tag."""
    path = Path(value)
    epoch_tag = _resolve_epoch_tag(path)
    if path.suffix == '.pt':
        tag = path.parent.name
    elif path.exists() and path.is_dir():
        tag = path.name
    else:
        tag = path.stem if path.suffix else path.name

    tag = strip_split_suffix(strip_timestamp_prefix(tag))
    return tag, epoch_tag


def _resolve_test_tag(value) -> str:
    """Resolve one test/cache path or plain tag to its logical tag."""
    path = Path(value)
    tag = path.stem if path.suffix else path.name
    return strip_split_suffix(tag)


def get_test_long_tag(model_ref, test_ref, *, include_epoch=True) -> str:
    """ Resolve one compact shared base tag for test/eval artifacts.
    ToDo: generalize for any epoch, with special tag for best one"""

    model_name, epoch_tag = _resolve_model_parts(model_ref)
    test_name = _resolve_test_tag(test_ref)
    same_tag = (test_name == model_name)

    model_specs, _ = _parse_specs(model_name)
    test_specs, _ = _parse_specs(test_name)
    common_specs = model_specs or test_specs
    if common_specs and common_specs not in model_name:
        model_name = f"{model_name}_{common_specs}"

    if same_tag:
        test_name = ''
    else:
        test_name = _strip_same_specs(test_name, common_specs if test_specs == common_specs else None)
    if test_name == model_name:
        test_name = ''

    parts = [model_name]
    if include_epoch and epoch_tag is not None:
        parts.append(epoch_tag)
    if test_name:
        parts.append(test_name)
    return '_'.join(parts)


def get_test_short_tag(test_ref) -> str:
    """ Resolve one compact short tag for ROC naming and titles."""
    tag = _resolve_test_tag(test_ref)
    tag = re.sub(r'(?:^|_)(?:ft\d+|\d+ft)(?=_|$)', '_', tag)
    tag = re.sub(r'(?:^|_)(?:w\d+(?:o\d+)?-\d+(?:o\d+)?)(?=_|$)', '_', tag)
    tag = re.sub(r'__+', '_', tag).strip('_')
    return tag or 'test'


def get_test_title_lines(model_ref, test_ref) -> tuple[str, str]:
    """Return two short title lines: model tag and reduced test tag."""
    model_tag, _ = _resolve_model_parts(model_ref)
    return model_tag or 'model', get_test_short_tag(test_ref)


# def build_test_artifact_name(model_ref, test_ref, kind, *, unit=None, short=False, include_best_epoch=True)-> str:
def get_exporting_name(model_ref, test_ref, kind, *, unit='NA', short=False, **kwargs)-> str:
    """Build one canonical export stem from model/test refs and export kind."""

    base_tag = (get_test_short_tag(test_ref) if short else
                get_test_long_tag(model_ref, test_ref, include_epoch=kwargs.get('include_epoch', True)))

    if kind == 'roc':
        return f'ROC_{base_tag}_{unit}'
    if kind == 'summary':
        if not unit:
            raise ValueError('summary export name requires unit')
        if unit == 'stream':
            return f'{base_tag}_reports'
        return f'{base_tag}_{unit}-summary'
    if kind == 'events':
        return f'{base_tag}_events'
    if kind == 'timeline':
        return f'{base_tag}_stream-tst'
    if kind == 'raw':
        return f'{base_tag}_{unit}-tst'
    raise ValueError(f"Unsupported artifact kind: {kind}")
