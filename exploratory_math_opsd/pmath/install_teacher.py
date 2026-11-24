"""Install the opt-in target at the archived trainer's fixed-teacher forward."""
from common import ROOT
path=ROOT/'source/opsd_trainer.py'
old='''            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]'''
new='''            if os.environ.get("OPSD_PROJECTOR"):
                from common import projected_logits
                actor = self.accelerator.unwrap_model(model)
                if not self.fixed_teacher or not is_peft_model(actor):
                    raise RuntimeError("P requires a fixed unadapted base")
                if not hasattr(self, "_math_projection"):
                    self._math_projection = torch.load(os.environ["OPSD_PROJECTOR"],
                        map_location=inputs["student_input_ids"].device, weights_only=True)
                # Unwrap before disabling: DeepSpeed may wrap the PEFT model.
                with actor.disable_adapter():
                    teacher_logits = projected_logits(actor.get_base_model(),
                        inputs["student_input_ids"], inputs["student_attention_mask"],
                        student_prompt_len - 1, self._math_projection)
                outputs_teacher = None
            else:
'''+ '\n'.join('    '+line if line else '' for line in old.splitlines())
s=path.read_text()
assert 'import os' in s
if new not in s:
    assert s.count(old)==1
    path.write_text(s.replace(old,new))
