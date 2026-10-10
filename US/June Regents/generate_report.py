import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':   './US/June Regents/US June Regents Data.xlsx',
    'title':  'US History Regents — June 2026',
    'output': './US/June Regents/us_june_regents_report.html',
    'class_colors': {
        'Bermejo':  lib.C['blue'],
        'Dushin':   lib.C['orange'],
        'Kovelsky': lib.C['teal'],
        'Antaky':    lib.C['red'],
    },
})
