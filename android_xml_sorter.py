#!/usr/bin/env python3
"""
Android XML resource sorter.

Sorts direct child resource elements inside <resources> by their `name`
attribute (case-insensitive, ascending). Resource element contents are moved
as raw text and are not parsed/reformatted/re-serialized.

Comments immediately before a resource move with that resource. A blank line
is inserted whenever the first character of the resource name changes.

Usage:
    python android_xml_sorter.py path/to/strings.xml
    python android_xml_sorter.py path/to/values --recursive
    python android_xml_sorter.py path/to/strings.xml --dry-run
    python android_xml_sorter.py path/to/strings.xml --no-backup
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from xml.parsers import expat


class SortError(Exception):
    pass


_NAME_RE = re.compile(
    rb"""(?:^|\s)name\s*=\s*(?P<q>["'])(?P<name>.*?)(?P=q)""",
    re.DOTALL,
)


def _find_tag_end(data: bytes, start: int) -> int:
    """Return the index just after a markup token, respecting quoted '>'."""
    quote = None
    i = start
    while i < len(data):
        c = data[i]
        if quote is not None:
            if c == quote:
                quote = None
        elif c in (ord('"'), ord("'")):
            quote = c
        elif c == ord(">"):
            return i + 1
        i += 1
    raise SortError("Unterminated XML markup.")


def _tokens(data: bytes):
    """
    Yield (kind, start, end, tag_name, self_closing) for markup tokens.
    Text between tokens is intentionally ignored.
    """
    i = 0
    n = len(data)

    while i < n:
        lt = data.find(b"<", i)
        if lt < 0:
            break

        if data.startswith(b"<!--", lt):
            end_marker = data.find(b"-->", lt + 4)
            if end_marker < 0:
                raise SortError("Unterminated XML comment.")
            end = end_marker + 3
            yield ("comment", lt, end, None, False)
            i = end
            continue

        if data.startswith(b"<![CDATA[", lt):
            end_marker = data.find(b"]]>", lt + 9)
            if end_marker < 0:
                raise SortError("Unterminated CDATA section.")
            end = end_marker + 3
            yield ("other", lt, end, None, False)
            i = end
            continue

        if data.startswith(b"<?", lt):
            end_marker = data.find(b"?>", lt + 2)
            if end_marker < 0:
                raise SortError("Unterminated processing instruction.")
            end = end_marker + 2
            yield ("other", lt, end, None, False)
            i = end
            continue

        if data.startswith(b"<!", lt):
            # Includes a DOCTYPE. Account for quoted text and internal subsets.
            quote = None
            bracket_depth = 0
            j = lt + 2
            while j < n:
                c = data[j]
                if quote is not None:
                    if c == quote:
                        quote = None
                elif c in (ord('"'), ord("'")):
                    quote = c
                elif c == ord("["):
                    bracket_depth += 1
                elif c == ord("]") and bracket_depth:
                    bracket_depth -= 1
                elif c == ord(">") and bracket_depth == 0:
                    break
                j += 1
            if j >= n:
                raise SortError("Unterminated declaration.")
            end = j + 1
            yield ("other", lt, end, None, False)
            i = end
            continue

        end = _find_tag_end(data, lt + 1)
        body = data[lt + 1:end - 1].strip()
        if body.startswith(b"/"):
            name = body[1:].split(None, 1)[0].rstrip(b"/>")
            yield ("end", lt, end, name, False)
        else:
            self_closing = body.endswith(b"/")
            name = body.split(None, 1)[0].rstrip(b"/>")
            yield ("start", lt, end, name, self_closing)
        i = end


def _resource_blocks(data: bytes):
    """
    Locate the <resources> root and return its prefix, suffix, and sortable
    direct-child blocks. Each block is (name, raw_prefix_comments, raw_element).
    """
    # Validate XML before making any changes.
    parser = expat.ParserCreate()
    try:
        parser.Parse(data, True)
    except expat.ExpatError as exc:
        raise SortError(f"XML is not well-formed: {exc}") from exc

    depth = 0
    root_seen = False
    root_start_end = None
    root_end_start = None
    child_start = None
    child_tag = None
    child_depth = 0
    pending = b""
    blocks = []
    last_end = None

    for kind, start, end, tag_name, self_closing in _tokens(data):
        if kind == "comment" and depth == 1 and child_start is None:
            # Keep comment and its preceding whitespace attached to the next item.
            continue

        if kind == "start":
            if not root_seen:
                if tag_name != b"resources":
                    raise SortError("The document root must be <resources>.")
                root_seen = True
                root_start_end = end
                depth = 1
                last_end = end
                continue

            if depth == 1:
                child_start = start
                child_tag = tag_name
                child_depth = 2
                depth = 2
                if self_closing:
                    element_end = end
                    raw_element = data[child_start:element_end]
                    match = _NAME_RE.search(data[child_start:end])
                    if not match:
                        raise SortError(
                            f"Resource <{child_tag.decode(errors='replace')}> "
                            "has no name attribute; refusing to guess its position."
                        )
                    name = match.group("name").decode("utf-8", errors="strict")
                    gap = data[last_end:child_start]
                    comments = _extract_comments(gap)
                    blocks.append((name, comments, raw_element))
                    last_end = element_end
                    child_start = None
                    child_tag = None
                    depth = 1
                continue

            # A nested self-closing tag has no matching end token.
            if not self_closing:
                depth += 1
            continue

        if kind == "end":
            if depth == 2 and child_start is not None:
                # This closes the direct child element.
                element_end = end
                raw_element = data[child_start:element_end]
                opening_end = _find_tag_end(data, child_start + 1)
                match = _NAME_RE.search(data[child_start:opening_end])
                if not match:
                    raise SortError(
                        f"Resource <{child_tag.decode(errors='replace')}> "
                        "has no name attribute; refusing to guess its position."
                    )
                name = match.group("name").decode("utf-8", errors="strict")
                gap = data[last_end:child_start]
                comments = _extract_comments(gap)
                blocks.append((name, comments, raw_element))
                last_end = element_end
                child_start = None
                child_tag = None
                depth = 1
            elif depth == 2:
                # Closing a nested element inside a resource.
                depth = 1
            elif depth > 2:
                depth -= 1
            elif depth == 1:
                root_end_start = start
                depth = 0
                break

    if root_start_end is None or root_end_start is None:
        raise SortError("Could not locate a complete <resources> root element.")

    prefix = data[:root_start_end]
    suffix = data[last_end:root_end_start] + data[root_end_start:]
    return prefix, suffix, blocks


def _extract_comments(gap: bytes) -> bytes:
    """Retain comments from the inter-resource gap, excluding separator whitespace."""
    found = []
    for kind, start, end, _tag, _self in _tokens(gap):
        if kind == "comment":
            found.append(gap[start:end])
    if not found:
        return b""
    # Comments themselves remain byte-for-byte identical. Use the indentation
    # from the original gap for readable placement.
    indent_match = re.search(rb"(?:^|\n)([ \t]*)<!--", gap)
    indent = indent_match.group(1) if indent_match else b""
    return b"\n".join(indent + c for c in found) + b"\n"


def _sort_bytes(data: bytes) -> tuple[bytes, int]:
    prefix, suffix, blocks = _resource_blocks(data)
    if not blocks:
        raise SortError("No named resource elements found inside <resources>.")

    blocks.sort(key=lambda item: (item[0].casefold(), item[0]))

    # Infer the file's newline style and resource indentation.
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    first_element = blocks[0][2]
    indent_match = re.search(rb"(?:^|\r?\n)([ \t]*)<", first_element)
    indent = indent_match.group(1) if indent_match else b"  "

    output = bytearray(prefix)
    # Ensure the root opening tag and first resource are separated by one newline.
    if output and not output.endswith((b"\n", b"\r")):
        output.extend(newline)

    previous_initial = None
    for index, (name, comments, element) in enumerate(blocks):
        initial = name[0].casefold() if name else ""
        if index:
            output.extend(newline)
            if initial != previous_initial:
                output.extend(newline)

        if comments:
            output.extend(comments.replace(b"\r\n", b"\n").replace(b"\n", newline))
            # Comments already end in a newline.
        output.extend(element)
        previous_initial = initial

    # Preserve the original trailing content before </resources>.
    if suffix and not suffix.startswith((b"\n", b"\r")):
        output.extend(newline)
    output.extend(suffix)
    return bytes(output), len(blocks)


def _collect_files(paths, recursive: bool):
    result = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = path.rglob("*.xml") if recursive else path.glob("*.xml")
            result.extend(p for p in candidates if p.is_file())
        elif path.is_file():
            result.append(path)
        else:
            print(f"Skip (not found): {path}", file=sys.stderr)
    # Stable de-duplication.
    unique = []
    seen = set()
    for path in result:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main():
    ap = argparse.ArgumentParser(
        description="Sort Android XML resource elements by name, ascending."
    )
    ap.add_argument("paths", nargs="+", help="XML file(s) or directory/directories")
    ap.add_argument(
        "--recursive", action="store_true",
        help="When a directory is supplied, include XML files in subdirectories."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing files."
    )
    ap.add_argument(
        "--no-backup", action="store_true",
        help="Do not create a .bak backup before changing a file."
    )
    args = ap.parse_args()

    files = _collect_files(args.paths, args.recursive)
    if not files:
        print("No XML files found.")
        return 1

    failures = 0
    for path in files:
        try:
            original = path.read_bytes()
            sorted_data, count = _sort_bytes(original)
            if sorted_data == original:
                print(f"Already sorted: {path} ({count} resources)")
                continue

            if args.dry_run:
                print(f"Would sort: {path} ({count} resources)")
                continue

            if not args.no_backup:
                backup = path.with_suffix(path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(path, backup)

            path.write_bytes(sorted_data)
            print(f"Sorted: {path} ({count} resources)")
        except (OSError, SortError, UnicodeDecodeError) as exc:
            failures += 1
            print(f"ERROR: {path}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
