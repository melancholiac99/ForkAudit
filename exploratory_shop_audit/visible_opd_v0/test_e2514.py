import os
import unittest
from unittest.mock import patch

from .e2512_native import native_rank
from .e25_layout import Layout
from .e2514_search import selection


class MigrationTests(unittest.TestCase):
    def test_early_probe_waits_for_the_bound_new_snapshot(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .e2514_early_probe import await_optimizer
        from exploratory_shop_audit.grpo_forced_audit.common import save
        with TemporaryDirectory() as tmp:
            path=Path(tmp)
            save(path/'target_manifest.json', dict(scorer_sha='frozen'))
            with self.assertRaises(TimeoutError):
                await_optimizer(path,18,timeout=0)
            for arm,step,sha in [('P',18,'frozen'),('R',17,'frozen'),('R',18,'other')]:
                save(path/'optimizer.json',dict(arm=arm,update=step,before_sha=sha))
                with self.assertRaises(AssertionError):
                    await_optimizer(path,18,timeout=0)
            save(path/'optimizer.json',dict(arm='R',update=18,before_sha='frozen'))
            await_optimizer(path,18,timeout=0)

    def test_early_and_regular_workers_score_the_rank_once(self):
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from threading import Event
        from . import e2514_early_probe as probe
        from exploratory_shop_audit.grpo_forced_audit.common import read,save
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/probe.SERIAL/'pilot/18/R'
            save(path/'target_manifest.json',dict(scorer_sha='frozen'))
            save(path.parents[1]/'17/R/optimizer.json',dict(after_sha='frozen'))
            ready=Event()
            def score(rank,layout,k,arms,before_new_snapshot):
                ready.set()
                before_new_snapshot(path)
                save(path/f'fp32_part_{rank}.json',dict(done=True))
            with patch.object(probe,'check_lane'),patch.object(probe,'RUN',Path(tmp)), \
                 patch(__package__+'.e25_probe.configure',return_value=0), \
                 patch(__package__+'.e25_measure.per_update',side_effect=score) as worker, \
                 ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(probe.measure_rank,18,0,True)
                self.assertTrue(ready.wait(3))
                second=pool.submit(probe.measure_rank,18,0,False)
                save(path/'optimizer.json',dict(arm='R',update=18,before_sha='frozen'))
                first.result(timeout=5)
                second.result(timeout=5)
                self.assertEqual(worker.call_count,1)
                self.assertTrue(read(path/'fp32_timing_0.json')['early'])

    def test_probe_gpu_allocation_unions_overlapping_launches(self):
        from datetime import datetime,timezone
        from .e2514_delivery import probe_allocation
        def stage(name,begin,end):
            return dict(phase=name,finished=datetime.fromtimestamp(end,timezone.utc).isoformat(),
                        elapsed=end-begin,physical_devices='0,1,2,3',update=18)
        regular=stage('per_update',100,150)
        peers=[stage('train',0,98),stage('collect',100,160)]
        for end,duration,training in [(130,140,88),(80,120,70)]:
            early=dict(start=10.,end=end,physical_devices='0,1,2,3',exit_code=0)
            data=probe_allocation(regular,peers,early)
            self.assertEqual(data['fp32_seconds'],duration)
            self.assertEqual(data['overlap_by_phase'],{'train':training,'collect':50})

    def test_terminal_label_needs_two_consecutive_training_breaches(self):
        from .e2514_delivery import stop_record
        rates=[(10,.9),(20,.8),(30,.79)]
        self.assertEqual(stop_record(.9,rates,True)['status'],'EARLY_STOP')
        self.assertEqual(stop_record(.9,rates[:2],False)['status'],'COMPLETED')
        with self.assertRaises(AssertionError):stop_record(.9,rates[:2],True)
        with self.assertRaises(AssertionError):stop_record(.9,rates,False)

    def test_historical_arg_counts_require_unambiguous_evidence(self):
        from .e2514_queries import archived_token_counts
        self.assertEqual(archived_token_counts(1,{'search':1},7,{(6,)}),
                         ([7],'archived_single_search_total'))
        self.assertIsNone(archived_token_counts(1,{'illegal':1},7,{(6,)})[0])
        self.assertEqual(archived_token_counts(2,{},7,{(2,4),(3,4)})[0],[3,4])
        self.assertIsNone(archived_token_counts(2,{},7,{(2,5),(3,4)})[0])

    def test_physical_rank_in_group(self):
        for start in (0,4):
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':','.join(str(start+r) for r in range(4))}):
                for r in range(4):
                    native_rank(r)
                    self.assertEqual(os.environ['SHOP_GPU_IDS'],str(start+r))
                    self.assertEqual(os.environ['SHOP_GPU_ID'],str(start+r))

    def test_frozen_projectors(self):
        l=Layout('formal_2512')
        self.assertEqual(l.projector_file(30,'P-frozen',2),l.projector_file(1,'P',2))
        self.assertNotEqual(l.projector_file(30,'P',2),l.projector_file(30,'P-frozen',2))

    def test_split_selection_and_strict_subset(self):
        split='T_opd_eval'
        counts={20:[2,2,1,0],30:[0,1,1,1]}
        def records(step):
            return {(split,g,r):{'reason':'gold_purchase' if r<n else 'wrong_purchase'}
                    for g,n in enumerate(counts[step]) for r in range(4)}
        with patch('exploratory_shop_audit.visible_opd_v0.e2514_search.original',records),patch('exploratory_shop_audit.visible_opd_v0.e2514_search.save'):
            selected=selection()
        self.assertEqual([(r['goal'],r['strict_subset']) for r in selected],[(0,True),(1,False)])
        self.assertEqual(selected[1]['good_reps'],[0,1])
        self.assertEqual(selected[1]['failed_reps'],[1,2,3])


class SearchReportTests(unittest.TestCase):
    def test_archive_extension_counts_source_occurrences_and_reuses_query_cells(self):
        from .e2514_search_extension import extend_tasks,query_set
        original=[dict(split='V',goal=9,good_reps=[0,2,3],failed_reps=[1],strict_subset=False,
            queries={'20':{'same':1},'30':{'own':1}},missing={'20':{'unrecovered':2},'30':{}})]
        recovered=[dict(step=20,split='V',goal=9,rep=rep,first_query=q,in_intervention_source=True)
                   for rep,q in ((2,'same'),(3,'novel'))]
        result=extend_tasks(original,recovered)
        self.assertEqual(result[0]['queries']['20'],{'same':2,'novel':1})
        self.assertEqual(result[0]['missing']['20'],{})
        self.assertEqual(query_set(result)-query_set(original),{('V',9,'novel')})
        self.assertEqual(original[0]['missing']['20'],{'unrecovered':2})
        with self.assertRaises(AssertionError):extend_tasks(original,[recovered[0],recovered[0]])

    def test_fresh_comparisons_and_nonpositive_fraction_denominator(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from . import e2514_swap_report as module
        from exploratory_shop_audit.grpo_forced_audit.common import save,read
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'evidence'
            tasks=[dict(split=s,goal=i,queries={'20':{'good':2},'30':{'own':1}},missing={'20':{},'30':{}},strict_subset=True)
                   for i,s in enumerate(('T_opd_eval','V'))]
            save(out/'intervention_tasks.json',tasks)
            save(out/'intervention_result_pages.json',[dict(split=t['split'],goal=t['goal'],query=q,contains_qualifying=q=='good')
                 for t in tasks for q in ('good','own')])
            for actor in (20,30):
                for rank in range(4):
                    path=root/'swap'/f'{actor:03}'/f'rank{rank}'
                    data=[]
                    for t in tasks if rank==0 else []:
                        for q in ('good','own'):
                            n={20:{'good':4,'own':2},30:{'good':2,'own':0}}[actor][q] if t['split']=='T_opd_eval' else 0
                            data.extend(dict(split=t['split'],goal=t['goal'],query=q,rep=r,reason='gold_purchase' if r<n else 'wrong_purchase') for r in range(4))
                    save(path/'part_0000.json',data);save(path/'complete.json',dict(jobs=len(data)))
            with patch.object(module,'OUT',out),patch.object(module,'DATA',root):
                module.report()
            reports=read(out/'search_swap_report.json')['summary']
            est=reports['T_opd_eval/strict/both_sources']['effects']
            self.assertEqual(est['P30_gain_good_minus_own']['estimate'],.5)
            self.assertEqual(est['P20_loss_own_minus_bad']['estimate'],.5)
            self.assertEqual(est['P30_recovery_fraction']['estimate'],.5)
            self.assertIsNone(reports['V/all/both_sources']['effects']['P30_recovery_fraction']['estimate'])
            for actor in (20,30):
                for rank in range(4):
                    sub=f'{actor:03}/rank{rank}'
                    records=read(root/'swap'/sub/'part_0000.json')
                    for name,query in (('swap','good'),('extra','own')):
                        part=[r for r in records if r['query']==query]
                        save(root/name/sub/'part_0000.json',part)
                        save(root/name/sub/'complete.json',dict(jobs=len(part)))
            module.report(out,[root/'swap',root/'extra'])
            self.assertEqual(read(out/'search_swap_report.json')['summary'],reports)

    def test_short_purchase_uses_visible_asin_and_native_query_string(self):
        from .e2514_queries import native_purchase_candidates
        q=dict(action='search[abc]',event='search',products=['111','222'])
        rows=list(native_purchase_candidates([[q]],dict(reason='gold_purchase',events={'buy':1,'search':1,'open_listed_product':1}),
                                              dict(asin='222',goal_options=[])))
        self.assertEqual({tuple(a['action'] for a in row) for row in rows},
            {('search[abc]','click[222]',cmd) for cmd in ('click[buy now]','click[Buy Now]')})
