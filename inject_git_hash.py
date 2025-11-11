#!/usr/bin/env python
"""
Script to inject git hash into version strings for development wheels.
This script modifies version strings in-place to include the git hash
in the format: X.Y.Z.dev0+gHASH

This should only be run for development wheels (S3 deployment), not
for release wheels (PyPI deployment).
"""
import os
import re
import subprocess
import sys


def get_git_hash():
    """Get the short git hash of the current commit."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True
        ).strip()
        return git_hash
    except subprocess.CalledProcessError as e:
        print(f"Error getting git hash: {e}", file=sys.stderr)
        sys.exit(1)


def inject_git_hash_in_file(filepath, search_pattern, replace_pattern):
    """
    Inject git hash into a version string in a file.
    
    Args:
        filepath: Path to the file to modify
        search_pattern: Regex pattern to find the version string
        replace_pattern: Replacement pattern with {git_hash} placeholder
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    git_hash = get_git_hash()
    
    # Replace the version string
    def replacer(match):
        # Get the version from the capture group
        # The version is in the second capture group (index 1)
        if len(match.groups()) >= 2:
            version = match.group(2)
            prefix = match.group(1)
        else:
            version = match.group(1)
            prefix = ""
        
        # Format the replacement
        result = replace_pattern.format(
            version=version,
            git_hash=git_hash
        )
        # Handle backreferences in the replacement pattern
        if r'\1' in replace_pattern:
            result = result.replace(r'\1', prefix)
        
        return result
    
    new_content = re.sub(
        search_pattern,
        replacer,
        content,
        flags=re.MULTILINE
    )
    
    if new_content == content:
        print(f"Warning: No changes made to {filepath}", file=sys.stderr)
        return False
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"Updated {filepath} with git hash: {git_hash}")
    return True


def main():
    """Inject git hash into all version files."""
    git_hash = get_git_hash()
    print(f"Injecting git hash: {git_hash}")
    
    # Files to update based on .bumpversion.cfg
    files_to_update = [
        {
            'path': 'setup.py',
            'search': r'(\s+version=)"([^"]+)"',
            'replace': r'\1"{version}+g{git_hash}"'
        },
        {
            'path': 'galpy/__init__.py',
            'search': r'(__version__ = )"([^"]+)"',
            'replace': r'\1"{version}+g{git_hash}"'
        },
        {
            'path': 'doc/source/conf.py',
            'search': r'(^version = )"([^"]+)"',
            'replace': r'\1"{version}+g{git_hash}"'
        },
        {
            'path': 'CITATION.cff',
            'search': r'(^version: )([^\n]+)',
            'replace': r'\1{version}+g{git_hash}'
        }
    ]
    
    success_count = 0
    for file_info in files_to_update:
        if inject_git_hash_in_file(
            file_info['path'],
            file_info['search'],
            file_info['replace']
        ):
            success_count += 1
    
    print(f"\nSuccessfully updated {success_count}/{len(files_to_update)} files")
    print(f"New version format: X.Y.Z.dev0+g{git_hash}")


if __name__ == "__main__":
    main()
