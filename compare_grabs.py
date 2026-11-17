"""Compare two recorded pickup attempts. Standard library only; never commands the arm."""
import argparse
import json
import math
from pathlib import Path
import zipfile

MAX_REPORT_BYTES = 8 * 1024 * 1024


def number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    result = [number(v) for v in value[:length]]
    return result if all(v is not None for v in result) else None


def difference(a, b):
    return [round(x-y, 4) for x,y in zip(a,b)] if a is not None and b is not None else None


def norm(value):
    return round(math.sqrt(sum(v*v for v in value)), 4) if value is not None else None


def summarize(report, source):
    if not isinstance(report, dict):
        raise ValueError(f'{source}: expected a JSON object')
    if isinstance(report.get('report'), dict):
        report = {**report, **report['report']}
    request = report.get('request') or {}
    wrist = request.get('wrist_alignment_report') or {}
    decision = request.get('height_decision') or {}
    fit = request.get('centre_fit') or {}
    descent = report.get('descent') or {}
    observations = wrist.get('observations') or []
    final_wrist = observations[-1] if observations else {}
    target = vector(report.get('target_cm'),3)
    if target is None:
        target = vector([request.get('x'),request.get('y'),request.get('height')],3)
    tip = vector(report.get('tip_cm'),3)  # settled, before closing; not last descent sample
    error = difference(tip,target)
    samples = descent.get('samples') or []
    first = vector(samples[0].get('tip_cm'),3) if samples else None
    last = vector(samples[-1].get('tip_cm'),3) if samples else None
    hover_tilt = number(final_wrist.get('hover_tilt_deg'))
    grab_tilt = number(report.get('tilt_deg'))
    if report.get('error'):
        outcome = 'FAILED'
    elif report.get('slipped') is True:
        outcome = 'SLIPPED (feedback)'
    elif report.get('holding') is True:
        outcome = 'HOLD DETECTED (feedback)'
    elif report.get('holding') is False:
        outcome = 'MISSED (feedback)'
    else:
        outcome = 'UNCONFIRMED'
    notes = []
    if not wrist:
        notes.append('No wrist trace in this report; this does not establish that the wrist was disabled.')
    elif wrist.get('status') != 'aligned':
        notes.append('Wrist alignment was not accepted. Inspect the recorded fallback reason and images.')
    else:
        notes.append('Wrist alignment was accepted at hover; this does not verify physical jaw centring during descent.')
    if tip is None:
        notes.append('Settled pre-close tip position is missing; last descent sample is not substituted for it.')
    if hover_tilt is not None and grab_tilt is not None and abs(grab_tilt-hover_tilt) > .01:
        notes.append('Recorded hover and pickup tilts differ. This is a lead to inspect, not proof of the miss cause.')
    return {
        'source':source, 'run_id':report.get('run_id'),
        'controller_build':report.get('controller_build'),
        'pickup_build':request.get('pickup_build') or wrist.get('pickup_build'),
        'object':wrist.get('text') or request.get('object_profile'),
        'outcome':outcome, 'visual_success':'not recorded', 'error':report.get('error'),
        'orientation':decision.get('orientation') or fit.get('orientation'),
        'height_source':request.get('height_source'), 'target_cm':target,
        'wrist_status':wrist.get('status'), 'fine_alignment':wrist.get('fine_alignment'),
        'last_wrist_offset_cm':number(final_wrist.get('offset_cm')),
        'wrist_look_count':len(observations) if wrist else None,
        'hover_tilt_deg':hover_tilt, 'pickup_tilt_deg':grab_tilt,
        'standing_xy_cm':vector(request.get('standing_xy_cm') or wrist.get('standing_xy_cm'),2),
        'wrist_selected_xy_cm':vector(wrist.get('final_xy_cm'),2),
        'settled_tip_before_close_cm':tip,
        'model_xy_error_cm':norm(error[:2]) if error else None,
        'model_z_error_cm':error[2] if error else None,
        'sample_xy_change_cm':difference(last[:2],first[:2]) if last and first else None,
        'descent_stop':descent.get('stop_reason'), 'notes':notes,
    }


def decode(data, source):
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError(f'{source}: report exceeds 8 MB')
    return summarize(json.loads(data), source)


def read_reports(path):
    path = Path(path)
    if path.suffix.lower() != '.zip':
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError(f'{path}: report exceeds 8 MB')
        return [decode(path.read_bytes(),str(path))]
    with zipfile.ZipFile(path) as archive:
        members = sorted((i for i in archive.infolist()
                          if '/grasp_runs/' in '/'+i.filename and i.filename.endswith('.json')
                          and '__MACOSX/' not in i.filename), key=lambda i:Path(i.filename).name)
        if not members:
            raise ValueError(f'{path}: no controller grasp reports in ZIP')
        reports = []
        for info in members[-2:]:
            if info.file_size > MAX_REPORT_BYTES:
                raise ValueError(f'{path}:{info.filename}: report exceeds 8 MB')
            # Read directly; no paths from the archive are extracted or executed.
            reports.append(decode(archive.read(info),f'{path}:{info.filename}'))
        return reports


def render(reports):
    a,b = reports
    def show(v):
        return 'unknown' if v is None else str(v)
    rows = [('Object','object'), ('Controller build','controller_build'), ('Pickup build','pickup_build'),
            ('Feedback outcome','outcome'), ('Orientation estimate','orientation'),
            ('Height source','height_source'), ('Requested XYZ cm','target_cm'),
            ('Wrist status','wrist_status'), ('Fine centring active','fine_alignment'),
            ('Last wrist offset cm','last_wrist_offset_cm'), ('Wrist observations','wrist_look_count'),
            ('Hover tilt deg','hover_tilt_deg'), ('Pickup tilt deg','pickup_tilt_deg'),
            ('Standing XY cm','standing_xy_cm'), ('Selected wrist XY cm','wrist_selected_xy_cm'),
            ('Settled pre-close XYZ cm','settled_tip_before_close_cm'),
            ('Model XY error cm','model_xy_error_cm'), ('Model Z error cm','model_z_error_cm'),
            ('First-to-last sample XY change cm','sample_xy_change_cm'), ('Descent stop','descent_stop'),
            ('Controller error','error')]
    lines = ['PICKUP COMPARISON', f'A: {a["source"]}', f'B: {b["source"]}', '',
             'A/B are recordings, not presumed success/failure labels.',
             'All position errors below use the saved model/encoders, not independent physical measurement.', '']
    for label,key in rows:
        lines.extend([label, '  A: '+show(a[key]), '  B: '+show(b[key])])
    for name,r in [('A',a),('B',b)]:
        lines.append('\n'+name+' interpretation limits:')
        lines.extend('  - '+s for s in r['notes'])
    if a['controller_build'] != b['controller_build'] or a['pickup_build'] != b['pickup_build']:
        lines.append('\nBuilds differ; do not attribute differences solely to object pose or calibration.')
    lines.append('\nCompare each run with its own camera images/video. No offsets or motion changes are applied.')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', help='two JSON reports / debug ZIPs, or one ZIP containing two reports')
    parser.add_argument('--out', type=Path, help='also save the comparison as a text file')
    args = parser.parse_args(argv)
    try:
        sources = args.sources
        if not sources:
            root = Path(__file__).resolve().parent/'config/local/grasp_runs'
            sources = sorted(root.glob('*.json'),key=lambda p:p.stat().st_mtime)[-2:]
            if len(sources) != 2:
                raise ValueError('Need two completed attempts. Pass two reports/debug ZIPs, or run after two attempts.')
        if len(sources) > 2:
            raise ValueError('Supply at most two files.')
        reports = read_reports(sources[0])[-2:] if len(sources)==1 else [read_reports(p)[-1] for p in sources]
        if len(reports) != 2:
            raise ValueError('Need two reports; one JSON file contains only one attempt.')
        text = render(reports)
        if args.out:
            args.out.write_text(text)
        print(text,end='')
    except (OSError,ValueError,KeyError,TypeError,AttributeError,zipfile.BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
