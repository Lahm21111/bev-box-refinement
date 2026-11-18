"""Evaluate only the three-way status gate for a generated example batch."""
import argparse
import os
from pathlib import Path
import sys

from model_runtime.gemini_box_decision import (build_request, call_api, create_run_directory,
                                             parse_answer, read_json, write_json)
from model_tools.actions import GATE_ACTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, default=Path('outputs/status_examples_10'))
    parser.add_argument('--output', type=Path, default=Path('outputs/gemini_status_10'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--start-case', type=int, default=1,
                        help='Start at this 1-based position in the manifest, skipping earlier cases')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'ultra_high'], default='ultra_high')
    parser.add_argument('--reference-mode', choices=['none', 'example', 'same-target'], default='none')
    parser.add_argument('--reference-example', type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.reference_mode == 'example' and not args.reference_example:
        parser.error('--reference-mode example requires --reference-example')
    if args.reference_example and args.reference_mode != 'example':
        parser.error('--reference-example is only used with --reference-mode example')
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not args.dry_run and not key:
        parser.error('Set GEMINI_API_KEY or GOOGLE_API_KEY, or use --dry-run')
    manifest = read_json(args.examples / 'manifest.json')
    if not 1 <= args.start_case <= len(manifest['cases']):
        parser.error('--start-case must be between 1 and the number of examples')
    output = create_run_directory(args.output)
    write_json(output / 'run.json', dict(mode='dry_run' if args.dry_run else 'live_api',
        decision_flow='gate_only', examples=str(args.examples.resolve()),
        reference_mode=args.reference_mode, start_case=args.start_case, model='gemini-decision-1'))
    results = []
    for case in manifest['cases'][args.start_case - 1:]:
        source = args.examples / case['id']
        mapping = read_json(source / 'views/pixel_mapping.json')
        reference = args.reference_example if args.reference_mode == 'example' else None
        if args.reference_mode == 'same-target':
            reference = args.examples / case['reference_case'] / 'views'
        payload = build_request(source / 'views', read_json(source / 'actions.json'), mapping,
            resolution=args.resolution, reference_example=reference,
            reference_mode='same-target' if args.reference_mode == 'same-target' else 'example', stage='gate')
        folder = output / case['id']
        folder.mkdir()
        write_json(folder / 'request.json', payload)
        (folder / 'instructions.txt').write_text(payload['questions']['next_action']['instructions'])
        if args.dry_run:
            print(f'{case["id"]}: gate only, 3 choices, request saved')
            continue
        # Evaluator labels and offsets are never included in payload.
        response = call_api(payload, key, args.timeout)
        write_json(folder / 'response.json', response)
        decision = parse_answer(response, GATE_ACTIONS)
        write_json(folder / 'decision.json', decision)
        result = dict(id=case['id'], category=case['category'], expected=case['expected_status'],
                      predicted=decision['choice'], confidence=decision['confidence'],
                      matches_expected=decision['choice'] == case['expected_status'])
        results.append(result)
        write_json(output / 'results.json', results)
        print(f'{case["id"]}: {decision["choice"]}, confidence={decision["confidence"]:.4f}')
    if results:
        matched = sum(r['matches_expected'] for r in results)
        write_json(output / 'summary.json', dict(count=len(results), matches_expected=matched,
            by_category={category: dict(count=sum(r['category'] == category for r in results),
                matches_expected=sum(r['category'] == category and r['matches_expected'] for r in results))
                for category in ['correct', 'small', 'large']}))
        print(f'Matches synthetic labels: {matched}/{len(results)}')
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        sys.exit(f'Error: {error}')
