__author__ = 'JB'
import os

def planet_detec_GPIepoch(inputDir,obj_list,spec_path_list = None,gather_detec_obj = None,outputDir = None,mute_error = True):

    inputDir = os.path.abspath(inputDir)
    compact_date=inputDir.split(os.path.sep)[-1].split("_")[0]

    err_list = []
    for obj in obj_list:
        iterating = True
        while iterating:
            if not mute_error:
                iterating = obj.initialize(inputDir=inputDir,compact_date=compact_date)
            else:
                try:
                    iterating = obj.initialize(inputDir=inputDir,compact_date=compact_date)
                except Exception as myErr:
                    err_list.append(myErr)
                    iterating = False
                    print("//!\\\\ "+obj.filename+" could NOT initialize in "+inputDir+". raised an Error.")

            if obj.spectrum_iter_available() and spec_path_list is not None:
                for spec_path in spec_path_list:
                    if not mute_error:
                        obj.init_new_spectrum(spec_path)
                        run(obj)
                    else:
                        try:
                            obj.init_new_spectrum(spec_path)
                            run(obj)
                        except Exception as myErr:
                            err_list.append(myErr)
                            print("//!\\\\ "+obj.filename+"with spectrum "+spec_path+" in "+inputDir+" raised an Error.")
            else:
                if not mute_error:
                    run(obj)
                else:
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