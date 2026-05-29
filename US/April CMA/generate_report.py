import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './US/April CMA/US April CMA Data.xlsx',
    'title':  'US History CMA — April 2026',
    'output': './US/April CMA/us_april_cma_report.html',
    'class_colors': {
        'Bermejo':  lib.C['blue'],
        'Dushin':   lib.C['orange'],
        'Kovelsky': lib.C['teal'],
    },
})
