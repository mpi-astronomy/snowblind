"""Print the CHANGES.md section for one version, for use as GitHub release notes.

The changelog follows Keep a Changelog: each release is a second-level
heading ``## [<version>] - <date>`` and the text up to the next
second-level heading is that release's notes.  Exit with an error when
the version has no section, so that a release cannot be published
without notes.
"""
import argparse
import re
import sys


def section(text, version):
    pattern = re.compile(r"^## \[(?P<version>[^\]]+)\][^\n]*\n(?P<body>.*?)(?=^## \[|\Z)", re.M | re.S)
    for match in pattern.finditer(text):
        if match.group("version") == version:
            return match.group("body").strip() + "\n"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="release version, as in the changelog heading and the git tag")
    parser.add_argument("--changelog", default="CHANGES.md")
    args = parser.parse_args()
    with open(args.changelog, encoding="utf-8") as handle:
        notes = section(handle.read(), args.version)
    if notes is None:
        sys.exit(f"{args.changelog} has no section for version {args.version}; add one before tagging the release")
    sys.stdout.write(notes)


if __name__ == "__main__":
    main()
