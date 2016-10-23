import subprocess


def generate_names(path):
    bash_cmd = "ls " + path + " | grep \.fts$"
    process = subprocess.Popen(["bash", "-c", bash_cmd],
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    file_name_list = output.split("\n")
    return file_name_prefix_list(file_name_list, path)


def file_name_prefix_list(old_list, prefix):
    new_list = []
    for name in old_list:
        new_list.append(prefix + name)
    return new_list
