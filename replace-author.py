#!/usr/bin/env python3

import sys

def replace_author_and_committer():
    old_name = b"Taha Kacmaz"
    old_email = b"tahakacmaz@example.com"
    new_name = b"noirrsw"
    new_email = b"noirrsw@gmail.com"

    for line in sys.stdin.buffer:
        if line.startswith(b"author "):
            if old_name in line or old_email in line:
                line = line.replace(old_name, new_name)
                line = line.replace(old_email, new_email)
        if line.startswith(b"committer "):
            if old_name in line or old_email in line:
                line = line.replace(old_name, new_name)
                line = line.replace(old_email, new_email)
        sys.stdout.buffer.write(line)

if __name__ == "__main__":
    replace_author_and_committer()