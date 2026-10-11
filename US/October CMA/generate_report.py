import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './US/October CMA/US October CMA Data.xlsx',
    'title':  'US History CMA — October 2026',
    'output': './US/October CMA/us_october_cma_report.html',
    'class_colors': {
        'Bermejo':        lib.C['blue'],
        'Dushin':         lib.C['orange'],
        'Klein / Cooper': lib.C['purple'],
    },
})
