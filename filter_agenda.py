#!/usr/bin/env python3
"""
Filter an agenda pickle by object properties.

Usage:
    # Extract only programs saved by goal_unproven_saver
    python filter_agenda.py agenda-saved.pkl -o saved_only.pkl --prop saved_from_goal_unproven

    # Extract only success programs
    python filter_agenda.py agenda.pkl -o success_only.pkl --prop verification_status=success
"""

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter agenda pickle by properties.")
    parser.add_argument('input', type=Path, help="Input pickle file")
    parser.add_argument('-o', '--output', type=Path, required=True, help="Output pickle file")
    parser.add_argument('--prop', action='append', required=True,
                        help="Property filter: 'key' (truthy) or 'key=value'. Can repeat.")

    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # Parse filters
    filters = []
    for p in args.prop:
        if '=' in p:
            key, value = p.split('=', 1)
            filters.append((key, value))
        else:
            filters.append((p, None))

    # Filter objects
    filtered = {}
    for path, obj in data['objects'].items():
        match = True
        for key, value in filters:
            prop_val = obj.properties.get(key)
            if value is None:
                if not prop_val:
                    match = False
            else:
                if str(prop_val) != value:
                    match = False
        if match:
            filtered[path] = obj

    # Write output with same structure
    out_data = {
        'objects': filtered,
        'tasks': data.get('tasks', {}),
        'status': data.get('status', {}),
        'clock': data.get('clock', 0),
    }

    with open(args.output, 'wb') as f:
        pickle.dump(out_data, f)

    print(f"Filtered {len(filtered)} / {len(data['objects'])} objects → {args.output}")


if __name__ == '__main__':
    main()
