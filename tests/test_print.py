import glob


def test_print(filesToCheck='/pyklip/**/.py'):
    files = glob.iglob(filesToCheck, recursive=True)
    multiline_comment = False
    bad_prints = {}
    for file in files:
        with open(file) as f:
            content = f.readlines()
        linecount = 1
        for line in content:
            if '\"\"\"' in line:
                multiline_comment = not multiline_comment
            if multiline_comment:
                linecount += 1
                continue;
            if 'print ' in line:
                if '#' in line:
                    hashtag = line.find('#')
                    line = line[:hashtag]
                if '\'' in line:
                    string_start = line.find('\'')
                    line_front = line[:string_start]
                    line = line[string_start+1:]

                    string_end = line.find('\'')
                    line_end = line[string_end:]
                    line = line_front + line_end

                if '\"' in line:
                    string_start = line.find('\"')
                    line_front = line[:string_start]
                    line = line[string_start+1:]

                    string_end = line.find('\"')
                    line_end = line[string_end:]
                    line = line_front + line_end
                    print(line)
                if 'print ' in line:
                    bad_prints.setdefault(file,[]).append(linecount)
            linecount += 1
    if(any(bad_prints)):
        error_message = ''
        for key in bad_prints.keys():
            error_message = error_message + '\tFile: ' + str(key) + '\n' + '\t\tLines: ' + str(bad_prints[key]) +'\n'
        raise SyntaxError('Bad print statements in:' + '\n' + error_message)

if __name__ == "__main__":
    test_print()