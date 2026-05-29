import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './Global/May Final/Global May Final Data.xlsx',
    'title':  'Global History Final — May 2026',
    'output': './Global/May Final/global_may_final_report.html',
    'class_colors': {
        'Sourial':  lib.C['blue'],
        'Dushin':   lib.C['orange'],
        'Kovelsky': lib.C['teal'],
    },
})
