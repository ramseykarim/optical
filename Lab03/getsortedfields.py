import subprocess


def generate_names(path):
    bash_cmd = "ls " + path + " | grep \.fts$"
    print ">>>>> " + bash_cmd + " <<<<<"
    process = subprocess.Popen(["bash", "-c", bash_cmd],
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    file_name_list = output.split("\n")
    return file_name_list

