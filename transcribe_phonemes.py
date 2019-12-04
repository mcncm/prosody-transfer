r"""Usage: first argument is a transcription program, second is a file suffix
that will be added to all the filenames. All subsequent arguments are file
names.

"""

import os
import sys
import subprocess
from multiprocessing import Pool
import multiprocessing

def transcribe_espeak(line):
    r"""Transcribes a line of text using espeak's ipa output.
    """
    print(line)
    # split line
    prefix, line_content = line.split('|')
    # transcribe
    completed_process = subprocess.run(['espeak', '--ipa', line], capture_output=True)
    return completed_process.stdout.decode('utf-8').strip()


def transcribe_file(f, method='espeak'):
    r"""Accepts a file object and returns a string of the transcribed file.
    Transcribes everything after the first appearance of the character `sep`.

    Do this with multiprocessing.
    """

    if method == 'espeak':
        transcribe_line = transcribe_espeak
    else:
        raise ValueError("This transcription method is not supported")

    p = Pool(multiprocessing.cpu_count())
    new_lines = p.map(transcribe_line, f.readlines())
    return new_lines


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: there should be a transcription method, output directory, and the files to consume")
        exit(1)

    transcription_method = sys.argv[1]
    out_dir = sys.argv[2]

    # put wherever the libspeak library is on your path. You might want
    # to change this
    os.environ['LD_LIBRARY_PATH'] += ':' + os.path.join(
        os.environ['HOME'], '.local', 'lib')

    # add the executable directory to the path. You might want to change this, too.
    os.environ['PATH'] += ':' + os.path.join(
        os.environ['HOME'], '.local', 'bin')

    for filename in sys.argv[3:]:
        with open(filename, 'r') as f:
            try:
                new_contents = transcribe_file(f, transcription_method)
            except ValueError as e:
                print(e)
                exit(2)

        with open(os.path.join(out_dir, filename), 'w') as g:
            g.writelines(new_contents)
