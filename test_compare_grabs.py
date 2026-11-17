"""Offline report comparison must keep missing evidence distinct from zero error."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import zipfile

import compare_grabs as compare


def report():
    return {'run_id':'demo','controller_build':'descent-taper-v1','holding':False,
            'target_cm':[1.,25.,.5], 'tip_cm':[1.3,25.4,.6], 'tilt_deg':30.,
            'request':{'pickup_build':'fine-centering-v3.2','object_profile':'black cap',
                       'height_source':'cap_height_range_fallback',
                       'wrist_alignment_report':{'status':'aligned','fine_alignment':True,
                                                'observations':[{'offset_cm':.08,'hover_tilt_deg':25.}]}},
            'descent':{'samples':[{'tip_cm':[1.,25.,6.]},{'tip_cm':[1.2,25.3,.7]}],
                       'stop_reason':'planned_path_complete'}}


class ComparisonTests(unittest.TestCase):
    def test_model_error_and_tilt_difference_remain_explicit_evidence(self):
        result=compare.summarize(report(),'example.json')
        self.assertEqual(result['model_xy_error_cm'],.5)
        self.assertEqual(result['model_z_error_cm'],.1)
        self.assertEqual(result['sample_xy_change_cm'],[.2,.3])
        self.assertEqual(result['outcome'],'MISSED (feedback)')
        self.assertEqual(result['visual_success'],'not recorded')
        self.assertIn('not proof',' '.join(result['notes']))

    def test_last_descent_sample_does_not_become_settled_closing_position(self):
        data=report(); del data['tip_cm']
        result=compare.summarize(data,'older.json')
        self.assertIsNone(result['settled_tip_before_close_cm'])
        self.assertIsNone(result['model_xy_error_cm'])
        self.assertIsNone(result['model_z_error_cm'])

    def test_missing_wrist_trace_does_not_mean_disabled(self):
        data=report(); data.pop('request')
        result=compare.summarize(data,'legacy.json')
        self.assertIsNone(result['wrist_status'])
        self.assertIsNone(result['fine_alignment'])
        self.assertIn('does not establish',' '.join(result['notes']))

    def test_unknown_and_nonfinite_values_never_become_zero_error(self):
        data=report(); data['tip_cm']=[1.,float('nan'),.5]
        data['holding']='true'
        result=compare.summarize(data,'invalid.json')
        self.assertIsNone(result['model_xy_error_cm'])
        self.assertEqual(result['outcome'],'UNCONFIRMED')

    def test_failure_and_slip_override_a_positive_hold(self):
        data=report(); data.update(holding=True,slipped=True)
        self.assertEqual(compare.summarize(data,'slip.json')['outcome'],'SLIPPED (feedback)')
        data['error']='lift failed'
        self.assertEqual(compare.summarize(data,'error.json')['outcome'],'FAILED')

    def test_comparison_marks_mixed_builds_and_keeps_valid_zero(self):
        a=report(); a['tip_cm']=list(a['target_cm'])
        b=copy.deepcopy(a); b['controller_build']='old-controller'
        summaries=[compare.summarize(a,'a.json'),compare.summarize(b,'b.json')]
        self.assertEqual(summaries[0]['model_xy_error_cm'],0.)
        text=compare.render(summaries)
        self.assertIn('Builds differ',text)
        self.assertIn('not presumed success/failure',text)

    def test_zip_comparison_reads_latest_reports_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bundle=root/'debug.zip'
            with zipfile.ZipFile(bundle,'w') as z:
                for i in (3,1,2):
                    data=report(); data['run_id']=str(i)
                    z.writestr(f'robot/config/local/grasp_runs/20261116_{i}.json',json.dumps(data))
                z.writestr('../../unwanted.txt','do not extract')
            output=io.StringIO()
            with redirect_stdout(output):
                compare.main([str(bundle)])
            self.assertEqual([r['run_id'] for r in compare.read_reports(bundle)],['2','3'])
            self.assertEqual(list(root.iterdir()),[bundle])
            self.assertIn('PICKUP COMPARISON',output.getvalue())

    def test_collector_includes_comparison_without_robot_or_network(self):
        import collect_arm_debug as collector
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            runs=root/'config/local/grasp_runs'; runs.mkdir(parents=True)
            for i in (1,2):
                (runs/f'run{i}.json').write_text(json.dumps(report()))
            output=root/'debug.zip'
            with patch.object(collector,'__file__',str(root/'collect_arm_debug.py')), \
                    patch.object(collector,'run_readonly',return_value=('offline fixture','')), \
                    patch.object(collector,'urlopen',side_effect=OSError('offline fixture')), \
                    patch('sys.argv',['collect_arm_debug.py','--output',str(output)]), \
                    redirect_stdout(io.StringIO()):
                collector.main()
            with zipfile.ZipFile(output) as z:
                self.assertIn('grasp_comparison.txt',z.namelist())
                self.assertIn('Model XY error cm',z.read('grasp_comparison.txt').decode())


if __name__ == '__main__':
    unittest.main()
