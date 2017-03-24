import glob
    """
    Tests ADI reduction with fakes injected at certain position angles

    Also tests lite mode

    Args:
        filelist: if not None, supply files to test on. Otherwise use standard beta pic data
    """

def test_print(filesToCheck='/pyklip/**/.py'):
    """
    Tests the entire pyklip directory for bad print statements in python files that would break if run on python 3.
    If there is ever a "print " as you would normally print in python 2, it will throw a syntax error with all the files and lines that the bad print statements were found in.
    Files should be written using python 3's print() function. 

    Cases it checks:
    Triple quote multiline comments  
    inline comments (#print)
    single quote character strings ('print ')
    double quote strings ("print ")

    Args: String of location and type of files to check. 
    Returns: A SyntaxError if there exist any bad print statements, otherwise nothing. 
    """

    #gathers all python files in pyklip directory recursively
    files = glob.iglob(filesToCheck, recursive=True)
    #used to check multiline comments such as doc strings. While multiline_comment is true it simply skips over the lines checking for the end before skipping.
    multiline_comment = False
    #dictionary to hold all bad prints so user can find and fix all bad print. (Key, Value) = ((string) File, (list) Line)
    bad_prints = {}

    for file in files:
        with open(file) as f:
            content = f.readlines()
        linecount = 1
        for line in content:
            #Checking for multiline comments before skipping. 
            if '\"\"\"' in line:
                multiline_comment = not multiline_comment
            if multiline_comment:
                linecount += 1
                continue;
            #If there exists a potential bad print check line, otherwise skip
            if 'print ' in line:
                #remove the inline comment part of line if it exists
                if '#' in line:
                    hashtag = line.find('#')
                    line = line[:hashtag]
                #remove the single quote string part of line if it exists
                if '\'' in line:
                    string_start = line.find('\'')
                    line_front = line[:string_start]
                    line = line[string_start+1:]

                    string_end = line.find('\'')
                    line_end = line[string_end:]
                    line = line_front + line_end
                #remove the double quote part of line if it exists
                if '\"' in line:
                    string_start = line.find('\"')
                    line_front = line[:string_start]
                    line = line[string_start+1:]

                    string_end = line.find('\"')
                    line_end = line[string_end:]
                    line = line_front + line_end
                #check line if it still has a bad print after reducing acceptable conditions
                if 'print ' in line:
                    #adds the entry to the dictionary of bad prints. 
                    bad_prints.setdefault(file,[]).append(linecount)
            linecount += 1
    #if anything exists in the bad_print dictionary raise a Syntax Error.
    if(any(bad_prints)):
        error_message = ''
        for key in bad_prints.keys():
            error_message = error_message + '\tFile: ' + str(key) + '\n' + '\t\tLines: ' + str(bad_prints[key]) +'\n'
        raise SyntaxError('Bad print statements in:' + '\n' + error_message)

if __name__ == "__main__":
    test_print()