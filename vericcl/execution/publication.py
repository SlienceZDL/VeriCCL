"""Publish a small result surface while retaining the complete process record."""
import json
import os
import uuid
from zipfile import ZIP_DEFLATED, ZipFile


def _read(context, relative):
    path = context.artifact(relative)
    return json.loads(path.read_text()) if path.is_file() else None


def _export(context, source, destination):
    context.write_bytes(destination, context.artifact(source).read_bytes(), replace=True)


def _archive(context):
    temporary = context.artifact('tmp/details-' + uuid.uuid4().hex + '.zip')
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(context.artifact('process').rglob('*')):
                if path == temporary:
                    continue
                relative = path.relative_to(context.root)
                checked = context.artifact(relative)
                if checked.is_file():
                    archive.write(checked, str(relative))
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, context.artifact('details.zip'))
        descriptor = os.open(context.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def publish_results(context, result, summary_text):
    """Export final artifacts from their authoritative process references."""
    artifacts = {'process': 'process/', 'report': 'report.md', 'log': 'run.log', 'details': 'details.zip'}
    latest = _read(context, 'checkpoints/latest.json')
    checkpoint = _read(context, latest['state']) if latest else None
    if checkpoint is not None:
        context.write_json('checkpoint.json', checkpoint | {'source': latest}, replace=True)
        artifacts['checkpoint'] = 'checkpoint.json'
    best = _read(context, 'best/accepted.json')
    model_source = checkpoint.get('model') if checkpoint else None
    if best is not None and result['acceptance'] in {'accepted', 'best_retained'}:
        candidate = best['candidate']
        model_source = 'models/' + best['model_version'] + '.json'
        for key, destination in (('xml', 'schedule.xml'), ('sidecar', 'schedule.json')):
            _export(context, candidate[key], destination)
        artifacts.update(schedule='schedule.xml', schedule_data='schedule.json')
        trace = best.get('trace_causality')
        if trace is not None:
            _export(context, trace['path'], 'trace.json')
            artifacts['trace'] = 'trace.json'
        else:
            context.artifact('trace.json').unlink(missing_ok=True)
    else:
        # A resumed run may finish without a qualified result. Old exports
        # remain in process history; the top-level view follows this result.
        for filename in ('schedule.xml', 'schedule.json', 'trace.json'):
            context.artifact(filename).unlink(missing_ok=True)
    if model_source is None:
        models = sorted(context.artifact('models').glob('*.json'))
        models = [path for path in models if not path.name.endswith('-diff.json')]
        if len(models) == 1:
            model_source = str(models[0].relative_to(context.root))
    if model_source is not None:
        _export(context, model_source, 'model.json')
        artifacts['model'] = 'model.json'
    else:
        context.artifact('model.json').unlink(missing_ok=True)
    timing = _read(context, 'reports/timing.json') or (checkpoint or {}).get('budgets')
    sections = [summary_text]
    if best is not None and 'schedule' in artifacts:
        sections.append('采用调度的预计完成时间：' + str(best.get('t_pred_us')) + ' 微秒。\n'
                        '代价模型：' + best['model_version'] + '。\n')
        checks = best['candidate'].get('checks', {})
        if checks:
            sections.append('检查结果：\n' + '\n'.join('- ' + name + '：' + str(check.get('status', check))
                            for name, check in checks.items()) + '\n')
    if timing is not None:
        sections.append('阶段时间与轮数详见 run.json 的 timing。\n')
        context.update_metadata('timing', timing, replace=True)
    if result.get('optional_failures'):
        sections.append('运行记录：\n' + '\n'.join('- ' + value for value in result['optional_failures']) + '\n')
    sections.append('完整过程保存在 process/，归档副本为 details.zip。\n')
    context.write_bytes('report.md', context.redactor.text('\n'.join(sections)).encode('utf-8'), replace=True)
    context.update_metadata('result', result, replace=True)
    context.update_metadata('best', best, replace=True)
    context.update_metadata('artifacts', artifacts, replace=True)
    context.update_metadata('publication', {'complete': False}, replace=True)
    _archive(context)
    context.update_metadata('publication', {'complete': True}, replace=True)
