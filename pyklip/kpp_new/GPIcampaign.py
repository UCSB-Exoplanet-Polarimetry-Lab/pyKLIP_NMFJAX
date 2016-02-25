__author__ = 'JB'
import os

def planet_detec_GPIepoch(inputDir,obj_list,spec_path_list = None,gather_detec_obj = None,outputDir = None):

    inputDir = os.path.abspath(inputDir)
    compact_date=inputDir.split(os.path.sep)[-1].split("_")[0]

    err_list = []
    for obj in obj_list:
        init_out = [-1,0]
        while init_out[0]<init_out[1]:
            try:
                init_out = obj.initialize(inputDir=inputDir,compact_date=compact_date)
            except Exception as myErr:
                err_list.append(myErr)
                init_out = [0,0]
                print("//!\\\\ "+obj.filename+" could NOT initialize in "+inputDir+". raised an Error.")

            if obj.spectrum_iter_available() and spec_path_list is not None:
                for spec_path in spec_path_list:
                    obj.init_new_spectrum(spec_path)
                    try:
                        run(obj)
                    except Exception as myErr:
                        err_list.append(myErr)
                        print("//!\\\\ "+obj.filename+"with spectrum "+spec_path+" in "+inputDir+" raised an Error.")
            else:
                # run(obj)
                try:
                    run(obj)
                except Exception as myErr:
                    err_list.append(myErr)
                    print("//!\\\\ "+obj.filename+" in "+inputDir+" raised an Error.")
    return err_list


def run(obj):
    if not obj.check_existence():
        map = obj.calculate()
        obj.save()
    else:
        map = obj.load()

    return None