import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import report_lib as lib

lib.run({
    'file':         './Global/June Regents/Global June 2026 Regents Data.xlsx',
    'title':        'Global History Regents — June 2026',
    'output':       './Global/June Regents/global_june_regents_report.html',
    'show_classes': False,
})
