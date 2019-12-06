"""Strip all IPA prosodic symbols from a text write the output to a new file.
First arg is a directory in which to put all the files. Remaining args are the
files.

"""
import os
import sys

banned_chars = ['ˈ', 'ː', 'ˌ', '̩']

if __name__ == '__main__':
    out_dir = sys.argv[1]
    try:
        os.makedirs(out_dir)
    except OSError as e:
        pass

    for filepath in sys.argv[2:]:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        filtered_lines = [''.join([c for c in line if c not in banned_chars])
                          for line in lines]
        filename = os.path.basename(filepath)
        with open(os.path.join(out_dir, filename), 'w') as g:
            g.writelines(filtered_lines)
