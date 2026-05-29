import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './US/May Final/US May Final Data.xlsx',
    'title':  'US History Final — May 2026',
    'output': './US/May Final/us_may_final_report.html',
    'class_colors': {
        'Bermejo':  lib.C['blue'],
        'Dushin':   lib.C['orange'],
        'Kovelsky': lib.C['teal'],
    },
})
