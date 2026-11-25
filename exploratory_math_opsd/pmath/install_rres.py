from common import ROOT
p=ROOT/'source/opsd_trainer.py'
s=p.read_text()
anchor='            if os.environ.get("OPSD_PROJECTOR"):'
patch='''            if os.environ.get("OPSD_RRES_PROJECTOR"):
                from rres_teacher import residual_logits
                actor = self.accelerator.unwrap_model(model)
                if not self.fixed_teacher or not is_peft_model(actor):
                    raise RuntimeError("Rres requires a frozen initial-model teacher")
                if not hasattr(self, "_residual_projection"):
                    self._residual_projection = torch.load(os.environ["OPSD_RRES_PROJECTOR"],
                        map_location=inputs["student_input_ids"].device, weights_only=True)
                with actor.disable_adapter():
                    teacher_logits = residual_logits(actor.get_base_model(),inputs,
                        student_prompt_len,teacher_prompt_len,self._residual_projection)
                outputs_teacher = None
            elif os.environ.get("OPSD_PROJECTOR"):'''
if patch not in s:
    assert s.count(anchor)==1
    p.write_text(s.replace(anchor,patch))
