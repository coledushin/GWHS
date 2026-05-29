import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './Global/April CMA/Global April CMA Data.xlsx',
    'title':  'Global History CMA — April 2026',
    'output': './Global/April CMA/global_april_cma_report.html',
    'class_colors': {
        'Sourial':  lib.C['blue'],
        'Dushin':   lib.C['orange'],
        'Kovelsky': lib.C['teal'],
    },
})
