import subprocess


def generate_names(path):
    bash_cmd = "ls " + path + " | grep \.fts$"
    process = subprocess.Popen(bash_cmd.split(),
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    file_name_list = output.split("\n")
    return file_name_list

