#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

def get_current_version(file_path):
    content = file_path.read_text()
    match = re.search(r'version = "(\d+\.\d+\.\d+)"', content)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return match.group(1)

def bump_version(current_version, part):
    major, minor, patch = map(int, current_version.split('.'))
    if part == 'major':
        major += 1
        minor = 0
        patch = 0
    elif part == 'minor':
        minor += 1
        patch = 0
    elif part == 'patch':
        patch += 1
    return f"{major}.{minor}.{patch}"

def update_file(file_path, current_version, new_version, pattern):
    content = file_path.read_text()
    new_content = re.sub(
        pattern.format(re.escape(current_version)),
        pattern.format(new_version),
        content,
        count=1
    )
    if content == new_content:
        print(f"Warning: No changes made to {file_path}")
    file_path.write_text(new_content)
    print(f"Updated {file_path} from {current_version} to {new_version}")

def main():
    parser = argparse.ArgumentParser(description="Bump version of the package")
    parser.add_argument('part', choices=['major', 'minor', 'patch'], default='patch', nargs='?', help="Part of version to bump")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    init_path = project_root / "src/transformer_cloner/__init__.py"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)
    
    if not init_path.exists():
        print(f"Error: {init_path} not found")
        sys.exit(1)

    try:
        current_version = get_current_version(pyproject_path)
        new_version = bump_version(current_version, args.part)
        
        print(f"Bumping version: {current_version} -> {new_version}")
        
        # Update pyproject.toml
        update_file(pyproject_path, current_version, new_version, r'version = "{}"')
        
        # Update __init__.py
        update_file(init_path, current_version, new_version, r'__version__ = "{}"')
        
        print("\nVersion bumped successfully!")
        print("Don't forget to commit and push the changes.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
