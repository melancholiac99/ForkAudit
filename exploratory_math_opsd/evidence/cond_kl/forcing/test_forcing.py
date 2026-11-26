import unittest
from types import SimpleNamespace as NS
from budget import install_forcing

class Generator:
    offset=100
    def set_offset(self,x):self.offset=x
    def get_offset(self):return self.offset

class TestForcing(unittest.TestCase):
    def test_sequence_and_guards(self):
        req=NS(output_token_ids=[],sampling_params=NS(seed=1001,max_tokens=100,extra_args=dict(forcing_row=2,target_lengths=[45,28])))
        generator=Generator()
        runner=NS(use_async_scheduling=False,input_batch=NS(req_ids=['x'],generators={0:generator}),_to_list=lambda v:[[v]],
                  discard_request_indices=NS(np=[]),num_discarded_requests=0,requests={'x':req})
        install_forcing(NS(model_runner=runner),[9,10],[4,5,6])
        for _ in range(8):
            for expect in (4,5,6):
                result=runner._to_list(10)[0][0]
                self.assertEqual(result,expect);req.output_token_ids.append(result)
        self.assertEqual(runner._to_list(9),[[9]])
        self.assertEqual(len(runner.forcing_events[2,0]),8)
        self.assertEqual(generator.offset,36)
        runner.forcing_events.clear()
        req.output_token_ids=[0]*44
        self.assertEqual(runner._to_list(9),[[9]])
        req.sampling_params.max_tokens=46
        req.sampling_params.extra_args['target_lengths'][0]=46
        self.assertEqual(runner._to_list(9),[[9]])
if __name__=='__main__':unittest.main()
