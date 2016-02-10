__author__ = 'JB'
import os

def planet_detec_GPIepoch(inputDir,metric_obj_list,stat_obj_list = None,spec_path_list = None,gather_detec_obj = None,outputDir = None):

    inputDir = os.path.abspath(inputDir)
    compact_date=inputDir.split(os.path.sep)[-1].split("_")[0]

    for metricObj in metric_obj_list:
        metricObj.initialize(inputDir=inputDir,compact_date=compact_date)

        if metricObj.spectrum_iter_available() and spec_path_list is not None:
            for spec_path in spec_path_list:
                metricObj.init_new_spectrum(spec_path)
                run_metric_and_stat(metricObj,stat_obj_list = stat_obj_list)
        else:
            run_metric_and_stat(metricObj,stat_obj_list = stat_obj_list)



def run_metric_and_stat(metricObj,stat_obj_list = None):
    if not metricObj.check_existence():
        metricMap = metricObj.calculate()
        metricObj.save()
    else:
        metricMap = metricObj.load()

    return None