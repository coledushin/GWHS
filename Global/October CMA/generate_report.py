import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './Global/October CMA/Global October CMA Data.xlsx',
    'title':  'Global History CMA — October 2026',
    'output': './Global/October CMA/global_october_cma_report.html',
    'class_colors': {
        'Sourial': lib.C['blue'],
        'Dushin':  lib.C['orange'],
    },
})
