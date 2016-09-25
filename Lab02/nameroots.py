import subprocess
"""
This process unpacks the first half of the filenames
contained in SpectralData. The second half consists
of 5 numbers, with leading zeros. Most of the numbers
only go up to 100.
"""


def generate_roots():
    """
    Nothing fancy here, just call it.
    :return list of all filename roots
    """
    bash_cmd = "sh ./getnameroots.sh"
    process = subprocess.Popen(bash_cmd.split(),
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    file_name_list = output.split("\n")
    return file_name_list
