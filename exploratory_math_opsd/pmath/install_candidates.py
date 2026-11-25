from common import ROOT
p=ROOT/'source/opsd_trainer.py'
s=p.read_text()
def edit(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)
if 'OPSD_SCREEN_ARM' not in s:
    edit('        token_clip=None,\n','        token_clip=None,\n        position_weights=None,\n        zero_target_token=None,\n')
    edit('        # Per-token clipping:', '''        if zero_target_token is not None:
            # torch.kl_div(log_target=True) returns NaN for exp(-inf)*(-inf).
            # The forward KL limit at teacher probability zero is zero.
            jsd[...,zero_target_token] = 0.

        # Per-token clipping:''')
    edit('        # Apply reduction\n','''        if position_weights is not None:
            from rw_objective import weighted_sum
            assert reduction=="batchmean" and labels is not None
            return weighted_sum(jsd,position_weights[mask])/mask.sum()

        # Apply reduction
''')
    anchor='            if os.environ.get("OPSD_RRES_PROJECTOR"):'
    edit(anchor,'''            screen_arm=os.getenv("OPSD_SCREEN_ARM")
            position_weights=None
            zero_target_token=None
            if screen_arm in ("Rw","Rnt"):
                from rw_objective import unpredictability,remove_think
                actor=self.accelerator.unwrap_model(model)
                assert self.fixed_teacher and is_peft_model(actor)
                with actor.disable_adapter():
                    base=actor.get_base_model()
                    outputs_teacher=base(input_ids=inputs["teacher_input_ids"],attention_mask=inputs["teacher_attention_mask"],use_cache=False)
                    teacher_logits=outputs_teacher.logits[:,teacher_prompt_len-1:-1]
                    if screen_arm=="Rw":
                        if not hasattr(self,"_rw_projector"):
                            self._rw_projector=torch.load(os.environ["OPSD_SCREEN_PROJECTOR"],weights_only=True,map_location=teacher_logits.device)
                        position_weights=unpredictability(base,inputs,student_prompt_len,teacher_logits,shifted_labels,self._rw_projector)
                    else:
                        zero_target_token=self.processing_class.convert_tokens_to_ids("<think>")
                        assert self.processing_class.convert_ids_to_tokens(zero_target_token)=="<think>"
                        remove_think(teacher_logits,zero_target_token)
            elif os.environ.get("OPSD_RRES_PROJECTOR"):''')
    edit('                token_clip=self.jsd_token_clip,','                token_clip=self.jsd_token_clip,\n                position_weights=position_weights,zero_target_token=zero_target_token,')
    p.write_text(s)
